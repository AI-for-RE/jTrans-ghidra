# Prerequests
- Ghidra 11.3+ (headless, no GUI needed) and a JDK 21+
- Python 3.8+ with pyghidra, networkx, pyelftools, tqdm
- GNU binutils `strip` (or `llvm-strip`, see `STRIP` below)
```bash
python3 -m pip install pyghidra pyelftools networkx tqdm
export GHIDRA_INSTALL_DIR=/path/to/ghidra   # e.g. /opt/ghidra_11.3_PUBLIC
```

# Quick Start

## Directory Description
- dataset (original binaries)
- dataset_strip (temp directory for strip binary)
- extract (extracted feature)
- ghidra_proj (Ghidra projects created during analysis)
- log (processing log)
- util (scripts utilities)
    - base.py (binary process base class)
    - disasm.py (renders Ghidra disassembly in IDA's text format)
    - pairdata (pair the groudtruth for functions with different optimization)
- process.py (PyGhidra script for extrating features of binaries)
- playdata.py (play with the extracted features)
- run.py (parallel run)

# Usage
## Extracting features for binary similarity task
- copy all the compiled binaries with symbol table to dataset/
- run the following commands
```bash
python3 run.py            # -j N for parallelism, default 8
```
Each binary is stripped into `dataset_strip/` before analysis, so no symbol names leak
into the disassembly; function names and addresses come from the symbol table of the
unstripped copy. Set `STRIP=llvm-strip` if GNU `strip` is not your default, or pass
`--no-strip` to analyze the binaries as-is.

A single binary can also be processed on its own:
```bash
python3 process.py ./dataset_strip/foo.strip --unstrip-path ./dataset/foo
```

## Notes on the Ghidra port
These scripts used to run inside IDA Pro (`idat64 -S process.py`); they now run as
ordinary Python programs that drive Ghidra through PyGhidra. Two things are worth
knowing:

- **The disassembly text is still IDA-flavoured.** `readidadata.parse_asm`, the jump
  handling in `data.gen_funcstr` and the pretrained vocabulary in
  `jtrans_tokenizer/vocab.txt` are all built around IDA's syntax, so `util/disasm.py`
  rebuilds each instruction from Ghidra's structured operands and prints it IDA-style
  (`mov rax, qword ptr [rsp+var_20]`, `jz loc_401227`, `sub rsp, 128h`). Emitting
  Ghidra's native text instead would push nearly every token out of vocabulary.
- **The last feature slot changed.** The IDA version stored
  `binaryai.ida.get_func_feature()` there; the BinaryAI SDK is IDA-only, so
  `process.py` computes an attributed-CFG descriptor from Ghidra instead (per-block
  instruction statistics and edges). jTrans itself never reads this field.

## Use the extracted features
- Have a look at util/playdata.py
- There are two types of processed datasets, one for unsupervised learning (unpair_data) and another for supervised learning (pair_data), which are stored in .pickle files
- unpair data
    ```python
    unpair_data = {
        'foo': [
                0x400000, # function_addr
                ['sub rbp, rsp', 'ret'], # asm_list
                b"\x48\x29\xe5\xc3", # raw bytes
                cfg, # networkx DiGraph
                acfg_feature
            ],
        'bar': [
            ...
        ]
    }
    # cfg traverse node
    def traverse_cfg_node(self, cfg):
        for node in cfg.nodes():
            yield cfg.nodes[node]['asm'], cfg.nodes[node]['raw']

    # cfg create code
    def get_cfg(self, func):
        body = func.getBody()

        def get_attr(block):
            asm, raw = [], b""
            for code_unit in self.listing.getCodeUnits(block, True):
                asm.append(self.formatter.code_unit(code_unit, body))
                raw += self.get_bytes(code_unit)
            return asm, raw

        nx_graph = nx.DiGraph()
        blocks = self.block_model.getCodeBlocksContaining(body, self.monitor)
        while blocks.hasNext():
            block = blocks.next()
            # Make sure all nodes are added (including edge-less nodes)
            asm, raw = get_attr(block)
            node = block.getMinAddress().getOffset()
            nx_graph.add_node(node, asm=asm, raw=raw)

            for pred in edge_blocks(block.getSources(self.monitor), source=True):
                nx_graph.add_edge(pred, node)
            for succ in edge_blocks(block.getDestinations(self.monitor), source=False):
                nx_graph.add_edge(node, succ)
        return nx_graph
    ```
- pair data
    the pair data is organized by the groundtruth (paired functions compiled by diferent optimization)
    ```python
    pair_data = {
            'foo': [
                unpair_foo_O0, # unpair_func_foo_O0
                unpair_foo_O1, # unpair_func_foo_O1
                unpair_foo_O2, # unpair_func_foo_O2
                ...
            ],
            'bar': [
                ...
            ]
        }
    ```
