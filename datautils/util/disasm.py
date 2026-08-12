"""Render Ghidra disassembly in the IDA Pro text format expected by jTrans.

The extracted pickles are consumed by ``readidadata.parse_asm`` and by the jump-target
rewriting in ``data.gen_funcstr``, both of which are written against IDA's syntax, and
the pretrained vocabulary in ``jtrans_tokenizer/vocab.txt`` is built from IDA tokens.
Ghidra's native text differs in every detail those consumers care about::

    Ghidra                            IDA (what we emit)
    ------------------------------    ------------------------------
    MOV RAX,qword ptr [RSP + 0x118]   mov rax, qword ptr [rsp+var_20]
    SUB RSP,0x128                     sub rsp, 128h
    JZ 0x00401227                     jz loc_401227
    CALL 0x00423030                   call sub_423030
    MOV RAX,qword ptr FS:[0x28]       mov rax, qword ptr fs:[28h]

so instead of emitting ``instruction.toString()`` we rebuild the operand text from
Ghidra's structured operand representation (registers, scalars, addresses and
separator characters) and apply IDA's spelling to each piece.

The ``loc_`` spelling matters most: ``parse_operand`` turns a ``loc_``/``sub_`` jump
target into ``hex_<addr>``, which ``gen_funcstr`` then resolves against the basic-block
addresses of the CFG to produce jTrans' jump tokens.  Those addresses therefore have to
be the real block addresses, which is why jump targets are rendered from the address
itself rather than from Ghidra's ``LAB_`` label names.
"""

# Mnemonic groups used for the attributed-CFG features (see process.BinaryData).
TRANSFER_INSTS = {
    'mov', 'movabs', 'movzx', 'movsx', 'movsxd', 'movsd', 'movss', 'movaps', 'movdqa',
    'movdqu', 'movups', 'lea', 'push', 'pop', 'xchg', 'cmov', 'set', 'stos', 'lods',
    'movs', 'cdq', 'cdqe', 'cwde',
}
ARITHMETIC_INSTS = {
    'add', 'adc', 'sub', 'sbb', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg',
    'addsd', 'subsd', 'mulsd', 'divsd', 'addss', 'subss', 'mulss', 'divss',
}
LOGIC_INSTS = {
    'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror', 'rcl', 'rcr',
    'test', 'bt', 'bts', 'btr', 'bswap',
}

# IDA's data directives, keyed by item size.
_DATA_DIRECTIVES = {1: 'db', 2: 'dw', 4: 'dd', 8: 'dq', 16: 'xmmword'}

# Mnemonics Ghidra and IDA spell differently.  The right-hand spellings are the ones
# present in jtrans_tokenizer/vocab.txt; without this map they all tokenize as OOV.
MNEMONIC_ALIASES = {
    'ret': 'retn',                        # IDA calls the near return retn
    'jc': 'jb', 'jnc': 'jnb',             # carry-flag conditions are named after
    'setc': 'setb', 'setnc': 'setnb',     # below/not-below in IDA
    'cmovc': 'cmovb', 'cmovnc': 'cmovnb',
    'jpe': 'jp', 'jpo': 'jnp',
}

# Ghidra hangs x86 prefixes off the mnemonic ("CMPXCHG.LOCK"); IDA writes them in
# front as a separate word ("lock cmpxchg").
MNEMONIC_PREFIXES = {'lock', 'rep', 'repe', 'repz', 'repne', 'repnz'}


def ida_hex(value):
    """Format an integer the way IDA prints it: 0-9 decimal, everything else ``NNNh``."""
    if 0 <= value <= 9:
        return str(value)
    text = format(value, 'X')
    if text[0].isalpha():  # IDA prefixes a 0 so the token cannot look like a symbol
        text = '0' + text
    return text + 'h'


class IdaFormatter(object):
    """Formats Ghidra code units as IDA-style disassembly text.

    A single instance is reused for a whole program so the Java class lookups and the
    symbol queries stay cheap.
    """

    def __init__(self, program):
        # Imported lazily: the JVM only exists after pyghidra.start().
        from ghidra.program.model.address import Address
        from ghidra.program.model.lang import Register
        from ghidra.program.model.listing import Instruction
        from ghidra.program.model.scalar import Scalar
        from ghidra.program.model.symbol import SourceType

        self._Address = Address
        self._Instruction = Instruction
        self._Register = Register
        self._Scalar = Scalar
        self._default_source = SourceType.DEFAULT

        self.program = program
        self.listing = program.getListing()
        self.func_manager = program.getFunctionManager()
        self.symbol_table = program.getSymbolTable()
        self.memory = program.getMemory()
        # Width used to print negative immediates the way IDA does, as two's complement.
        self.word_mask = (1 << (program.getDefaultPointerSize() * 8)) - 1

    # ------------------------------------------------------------------ code units

    def code_unit(self, code_unit, func_body=None):
        """Format an instruction or a data item, mirroring ``idc.GetDisasm``."""
        if isinstance(code_unit, self._Instruction):
            return self.instruction(code_unit, func_body)
        return self.data(code_unit)

    def instruction(self, instr, func_body=None):
        mnemonic = self.mnemonic(instr)
        operands = []
        for index in range(instr.getNumOperands()):
            text = self.operand(instr, index, func_body)
            if text:
                operands.append(text)
        if not operands:
            return mnemonic
        return mnemonic + ' ' + ', '.join(operands)

    def mnemonic(self, instr):
        """Ghidra mnemonic -> IDA spelling, including any instruction prefix."""
        mnemonic = instr.getMnemonicString().lower()
        prefix = ''
        if '.' in mnemonic:
            base, _, suffix = mnemonic.rpartition('.')
            if suffix in MNEMONIC_PREFIXES:
                mnemonic, prefix = base, suffix + ' '
        return prefix + MNEMONIC_ALIASES.get(mnemonic, mnemonic)

    def data(self, data):
        """Format a data item inside a function body (jump tables, alignment, ...)."""
        directive = _DATA_DIRECTIVES.get(data.getLength(), 'db')
        try:
            value = data.getValue()
        except Exception:
            value = None
        if value is None:
            return directive
        if isinstance(value, self._Address):
            return '%s offset %s' % (directive, self.address_name(value))
        try:
            return '%s %s' % (directive, ida_hex(int(value)))
        except (TypeError, ValueError):
            return directive

    # -------------------------------------------------------------------- operands

    def operand(self, instr, index, func_body=None):
        """Rebuild one operand from Ghidra's structured representation.

        ``getDefaultOperandRepresentationList`` hands back the operand as a list of
        ``Register`` / ``Scalar`` / ``Address`` objects interleaved with the separator
        characters, e.g. ``qword ptr [RSP + 0x10]`` arrives as
        ``['q','w','o','r','d',' ','p','t','r',' ','[', RSP, ' ', '+', ' ', 0x10, ']']``.
        """
        parts = instr.getDefaultOperandRepresentationList(index)
        if parts is None:
            text = instr.getDefaultOperandRepresentation(index)
            return text.lower() if text else ''

        stack_name = self._stack_variable(instr, index)
        out = []
        depth = 0  # bracket nesting: IDA writes memory operands without spaces
        saw_register = False
        negate = False  # a '-' seen outside brackets, applying to the next scalar
        for part in parts:
            if isinstance(part, self._Register):
                if negate:
                    out.append('-')
                    negate = False
                saw_register = True
                out.append(part.getName().lower())
            elif isinstance(part, self._Scalar):
                value = part.getValue()
                if depth and stack_name:
                    out.append(stack_name)  # only the displacement becomes var_/arg_
                    stack_name = None
                elif depth and value == 1 and out and out[-1] == '*':
                    out.pop()  # IDA omits a scale of 1: "[rax+rax*1]" -> "[rax+rax]"
                elif not depth and (negate or value < 0):
                    # "OR RAX,-0x1" -> "or rax, 0FFFFFFFFFFFFFFFFh": IDA prints negative
                    # immediates as two's complement rather than with a sign.
                    if negate:
                        value = -value
                    out.append(ida_hex(value & self.word_mask))
                    negate = False
                elif value >= 0:
                    out.append(ida_hex(value))
                else:
                    # Ghidra prints "[RBP + -0x8]", IDA prints "[rbp-8h]".
                    if out and out[-1] == '+':
                        out[-1] = '-'
                    out.append(ida_hex(-value))
            elif isinstance(part, self._Address):
                if negate:
                    out.append('-')
                    negate = False
                out.append(self.address_name(part, func_body))
            else:
                char = str(part)
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                elif char == ' ' and depth:
                    continue  # "[RSP + 0x10]" -> "[rsp+10h]"
                elif char == '-' and not depth:
                    negate = True  # held back until we see what it applies to
                    continue
                out.append(char.lower() if len(char) == 1 else char)
        if negate:
            out.append('-')

        text = ''.join(out)
        if not saw_register and text.endswith(']') and '[' in text:
            # An absolute (or rip-relative, which Ghidra already resolves) memory
            # operand: Ghidra writes "qword ptr [0x004b8e88]", IDA names the target,
            # "qword ptr ds:unk_4B8E88".
            target = self._data_reference(instr, index)
            if target is not None:
                return text[:text.index('[')] + 'ds:' + self.address_name(target, func_body)
        return text

    def _stack_variable(self, instr, index):
        """IDA renders frame accesses as ``var_X``/``arg_X``; recover that from Ghidra refs."""
        for ref in instr.getReferencesFrom():
            if ref.getOperandIndex() != index or not ref.isStackReference():
                continue
            offset = ref.getStackOffset()
            return ('var_%X' % -offset) if offset < 0 else ('arg_%X' % offset)
        return None

    def _data_reference(self, instr, index):
        """Address of the datum this operand reads or writes, if Ghidra recorded one."""
        for ref in instr.getReferencesFrom():
            if ref.getOperandIndex() != index or ref.isStackReference():
                continue
            if ref.getReferenceType().isData():
                return ref.getToAddress()
        return None

    # --------------------------------------------------------------------- symbols

    def address_name(self, address, func_body=None):
        """Name an address operand the way IDA would.

        Targets inside the function under analysis become ``loc_<addr>`` so that
        ``data.gen_funcstr`` can map them back onto CFG basic blocks; everything else
        falls back to the real symbol name, or to IDA's ``sub_``/``loc_``/``unk_``
        placeholders when Ghidra only has an auto-generated name.
        """
        offset = address.getOffset()
        if func_body is not None and func_body.contains(address):
            return 'loc_%X' % offset

        func = self.func_manager.getFunctionAt(address)
        if func is not None:
            symbol = func.getSymbol()
            if symbol is None or symbol.getSource() == self._default_source:
                return 'sub_%X' % offset
            return func.getName()

        symbol = self.symbol_table.getPrimarySymbol(address)
        if symbol is not None and symbol.getSource() != self._default_source:
            return symbol.getName()

        block = self.memory.getBlock(address)
        if block is not None and block.isExecute():
            return 'loc_%X' % offset
        return 'unk_%X' % offset
