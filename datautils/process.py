#!/usr/bin/env python3
"""
Extract per-function features from a binary using Ghidra, via PyGhidra.

Usage::

    python3 process.py ./dataset_strip/foo.strip --unstrip-path ./dataset/foo
"""
import argparse
import os
import pickle
import sys

import networkx as nx
import pyghidra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.base import Binarybase
from util.disasm import ARITHMETIC_INSTS, LOGIC_INSTS, TRANSFER_INSTS, IdaFormatter

SAVEROOT = "./extract"      # dir of pickle files saved by Ghidra
DATAROOT = "./dataset"      # dir of binaries (not stripped)
PROJECTROOT = "./ghidra_proj"  # dir of Ghidra projects (replaces IDA's idb/)

# IDA skipped these segments; the Ghidra ELF loader uses the same section names, plus
# EXTERNAL for the synthetic block holding imported functions.
SKIP_BLOCKS = {'.plt', '.plt.got', '.plt.sec', '.init', '.fini', 'extern', 'EXTERNAL'}


def offset_of(address):
    return int(address.getOffset())


class BinaryData(Binarybase):
    def __init__(self, unstrip_path, flat_api):
        super(BinaryData, self).__init__(unstrip_path)

        from ghidra.program.model.block import BasicBlockModel
        from ghidra.util.task import TaskMonitor

        self.flat = flat_api
        self.program = flat_api.getCurrentProgram()
        self.listing = self.program.getListing()
        self.func_manager = self.program.getFunctionManager()
        self.memory = self.program.getMemory()
        self.monitor = TaskMonitor.DUMMY
        self.block_model = BasicBlockModel(self.program)
        self.formatter = IdaFormatter(self.program)
        self.fix_up()

    def fix_up(self):
        """Define a function at every address the symbol table knows about.

        Same purpose as the IDA version's ``create_insn`` + ``add_func``: auto-analysis
        of a stripped binary misses some functions, and those are exactly the ones we
        need, since the ground truth is keyed by symbol address.
        """
        with pyghidra.transaction(self.program, "jTrans fix_up"):
            for addr in self.addr2name:
                entry = self.flat.toAddr(addr)
                if entry is None or not self.memory.contains(entry):
                    continue
                if self.listing.getInstructionAt(entry) is None:
                    try:
                        self.flat.disassemble(entry)
                    except Exception as err:
                        print("[!] cannot disassemble %#x: %s" % (addr, err))
                        continue
                if self.func_manager.getFunctionAt(entry) is None:
                    try:
                        self.flat.createFunction(entry, None)
                    except Exception as err:
                        print("[!] cannot create function at %#x: %s" % (addr, err))

    @staticmethod
    def get_bytes(code_unit):
        """Java signed bytes -> Python bytes."""
        try:
            return bytes(b & 0xFF for b in code_unit.getBytes())
        except Exception:
            return b""

    def get_asm(self, func):
        body = func.getBody()
        return [self.formatter.code_unit(cu, body)
                for cu in self.listing.getCodeUnits(body, True)]

    def get_rawbytes(self, func):
        rawbytes_list = b""
        for code_unit in self.listing.getCodeUnits(func.getBody(), True):
            rawbytes_list += self.get_bytes(code_unit)
        return rawbytes_list

    def get_cfg(self, func):
        body = func.getBody()

        def get_attr(block):
            asm, raw = [], b""
            for code_unit in self.listing.getCodeUnits(block, True):
                asm.append(self.formatter.code_unit(code_unit, body))
                raw += self.get_bytes(code_unit)
            return asm, raw

        def edge_blocks(iterator, source):
            """Yield the neighbouring block of each non-call flow inside this function."""
            while iterator.hasNext():
                ref = iterator.next()
                flow = ref.getFlowType()
                if flow is not None and flow.isCall():
                    continue  # calls are not CFG edges, matching IDA's FlowChart
                neighbour = ref.getSourceBlock() if source else ref.getDestinationBlock()
                if neighbour is None:
                    # The block is not always cached on the reference; a source address
                    # also points into the middle of its block rather than at its start.
                    addr = ref.getSourceAddress() if source else ref.getDestinationAddress()
                    if addr is None:
                        continue
                    neighbour = self.block_model.getFirstCodeBlockContaining(addr, self.monitor)
                    if neighbour is None:
                        continue
                start = neighbour.getMinAddress()
                if start is None or not body.contains(start):
                    continue
                yield offset_of(start)

        nx_graph = nx.DiGraph()
        blocks = self.block_model.getCodeBlocksContaining(body, self.monitor)
        while blocks.hasNext():
            block = blocks.next()
            start = block.getMinAddress()
            if start is None or not body.contains(start):
                continue
            # Make sure all nodes are added (including edge-less nodes)
            asm, raw = get_attr(block)
            node = offset_of(start)
            nx_graph.add_node(node, asm=asm, raw=raw)
            for pred in edge_blocks(block.getSources(self.monitor), source=True):
                nx_graph.add_edge(pred, node)
            for succ in edge_blocks(block.getDestinations(self.monitor), source=False):
                nx_graph.add_edge(node, succ)
        return nx_graph

    def get_func_feature(self, func, cfg):
        """Attributed CFG (ACFG) descriptor for the function.

        The IDA script filled this slot with ``binaryai.ida.get_func_feature()``.  The
        BinaryAI SDK only ships an IDA backend, so we compute an equivalent ACFG from
        Ghidra: per-basic-block instruction statistics plus the block's out-degree and
        number of descendants.  jTrans itself never reads this field -- it is kept so
        the pickle layout stays unchanged for anything else consuming the dataset.
        """
        nodes = {}
        for node in cfg.nodes():
            asm = cfg.nodes[node]['asm']
            mnemonics = [line.split(' ')[0] for line in asm if line]
            nodes[node] = {
                'n_inst': len(asm),
                'n_transfer': sum(1 for m in mnemonics if m in TRANSFER_INSTS),
                'n_call': sum(1 for m in mnemonics if m.startswith('call')),
                'n_arith': sum(1 for m in mnemonics if m in ARITHMETIC_INSTS),
                'n_logic': sum(1 for m in mnemonics if m in LOGIC_INSTS),
                'n_const': sum(1 for line in asm for token in line.split(' ')[1:]
                               if token.rstrip(',').endswith('h')),
                'n_offspring': len(nx.descendants(cfg, node)),
                'out_degree': int(cfg.out_degree(node)),
            }
        return {
            'arch': str(self.program.getLanguageID()),
            'entry': offset_of(func.getEntryPoint()),
            'n_node': cfg.number_of_nodes(),
            'n_edge': cfg.number_of_edges(),
            'nodes': nodes,
            'edges': list(cfg.edges()),
        }

    def extract_all(self):
        for func in self.func_manager.getFunctions(True):
            if func.isExternal() or func.isThunk():
                continue
            entry = func.getEntryPoint()
            block = self.memory.getBlock(entry)
            if block is not None and block.getName() in SKIP_BLOCKS:
                continue
            addr = offset_of(entry)
            print("[+] %s" % func.getName())
            asm_list = self.get_asm(func)
            rawbytes_list = self.get_rawbytes(func)
            cfg = self.get_cfg(func)
            feature = self.get_func_feature(func, cfg)
            yield (self.addr2name[addr], addr, asm_list, rawbytes_list, cfg, feature)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('binary', help='binary to analyze (normally the stripped copy)')
    parser.add_argument('--unstrip-path', default=None,
                        help='binary with symbols; defaults to DATAROOT/<name>')
    parser.add_argument('--dataroot', default=DATAROOT, help='dir of unstripped binaries')
    parser.add_argument('--saveroot', default=SAVEROOT, help='dir for the extracted pickles')
    parser.add_argument('--project-location', default=PROJECTROOT,
                        help='where the Ghidra project is created')
    parser.add_argument('--project-name', default=None,
                        help='Ghidra project name; defaults to the binary name')
    parser.add_argument('--ghidra-install-dir', default=None,
                        help='Ghidra installation; defaults to $GHIDRA_INSTALL_DIR')
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.exists(args.binary), f'{args.binary} not exists'
    assert os.path.exists(args.dataroot), f"DATAROOT {args.dataroot} does not exist"
    assert os.path.exists(args.saveroot), f"SAVEROOT {args.saveroot} does not exist"

    filename = os.path.basename(args.binary)
    if filename.endswith('.strip'):
        filename = filename[:-len('.strip')]
    unstrip_path = args.unstrip_path or os.path.join(args.dataroot, filename)

    install_dir = args.ghidra_install_dir or os.environ.get('GHIDRA_INSTALL_DIR')
    if not install_dir:
        sys.exit("[!] set GHIDRA_INSTALL_DIR (or pass --ghidra-install-dir)")
    os.makedirs(args.project_location, exist_ok=True)

    pyghidra.start(install_dir=install_dir)

    saved_dict = {}
    with pyghidra.open_program(args.binary,
                               project_location=args.project_location,
                               project_name=args.project_name or filename,
                               analyze=True) as flat_api:
        binary_data = BinaryData(unstrip_path, flat_api)
        for func_name, func, asm_list, rawbytes_list, cfg, feature in binary_data.extract_all():
            saved_dict[func_name] = [func, asm_list, rawbytes_list, cfg, feature]

    saved_path = os.path.join(args.saveroot, filename + "_extract.pkl")  # unpair data
    with open(saved_path, 'wb') as f:
        pickle.dump(saved_dict, f)
    print("[*] %d functions saved to %s" % (len(saved_dict), saved_path))


if __name__ == "__main__":
    main()
