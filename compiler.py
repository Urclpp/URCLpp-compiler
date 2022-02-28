from os.path import isfile
from sys import argv, stdout, stderr
from enum import Enum
from typing import List, OrderedDict

from dataclasses import dataclass


# TOKENS
class T(Enum):
    newLine = 'nln'
    word = 'wrd:'
    imm = 'opr_imm'
    reg = 'opr_reg'
    mem = 'opr_mem'
    port = 'opr_por'
    relative = 'opr_rel'
    label = 'opr_lab'
    macro = 'opr_mac'
    pointer = 'opr_poi'
    char = 'opr_cha'
    string = 'opr_str'
    sym_lpa = 'sym_lpa'
    sym_rpa = 'sym_rpa'
    sym_lbr = 'sym_lbr'
    sym_rbr = 'sym_rbr',
    sym_col = "sym_col",
    sym_gt = "sym_gt",
    sym_lt = "sym_lt",
    sym_geq = "sym_geq",
    sym_leq = "sum_leq",
    sym_equ = "sym_equ",
    sym_dif = "sym_dif",
    sym_and = "sym_and",
    sym_or = "sym_or",

    def __repr__(self) -> str:
        return self.value


# CONSTANTS
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
digits = '1234567890'
bases = 'oOxXbB'
indentation = ' \t\n'
symbols = {
    '(': T.sym_lpa,
    ')': T.sym_rpa,
    '[': T.sym_lbr,
    ']': T.sym_rbr,
    ':': T.sym_col,
    '<': T.sym_lt,
    '>': T.sym_gt,
    '>=': T.sym_geq,
    '<=': T.sym_leq,
    '!=': T.sym_dif,
    '==': T.sym_equ,
    '&&': T.sym_and,
    '||': T.sym_or,
}
op_precedence = {
    "sym_lt": 0,
    "sym_gt": 0,
    "sym_geq": 0,
    "sym_leq": 0,
    "sym_dif": 1,
    "sym_equ": 1,
    "sym_and": 2,
    "sym_or": 3,
    "sym_lpa": 5
}
operand_num = {  # number of operand a function expects
    # CORE
    'ADD': 3,
    'RSH': 2,
    'LOD': 2,
    'STR': 2,
    'BGE': 3,
    'NOR': 3,
    'IMM': 2,
    # I/O
    'IN': 2,
    'OUT': 2,
    # BASIC
    'SUB': 3,
    'JMP': 1,
    'MOV': 2,
    'NOP': 0,
    'LSH': 2,
    'INC': 2,
    'DEC': 2,
    'NEG': 2,
    'AND': 3,
    'OR': 3,
    'NOT': 2,
    'XOR': 3,
    'XNOR': 3,
    'NAND': 3,
    'BRE': 3,
    'BNE': 3,
    'BRL': 3,
    'BRG': 3,
    'BLE': 3,
    'BZR': 2,
    'BNZ': 2,
    'BRN': 2,
    'BRP': 2,
    'BEV': 2,
    'BOD': 2,
    'PSH': 2,
    'POP': 2,
    'CAL': 2,
    'RET': 0,
    'HLT': 0,
    'CPY': 2,
    'BRC': 3,
    'BNC': 3,
    # COMPLEX
    'MLT': 3,
    'DIV': 3,
    'MOD': 3,
    'BSR': 3,
    'BSL': 3,
    'SRS': 2,
    'BSS': 3,
    'SETE': 3,
    'SETNE': 3,
    'SETL': 3,
    'SETG': 3,
    'SETLE': 3,
    'SETGE': 3,
    'SETC': 3,
    'SETNC': 3,
    'LLOD': 3,
    'LSTR': 3,
    # URCLpp exclusive
    'END': 0,
    'EXIT': 0,
    'SKIP': 0,
    'IF': 1,
    'ELIF': 1,
    'ELSE': 0,
    'FOR': 2,
    'WHILE': 1,
    'SWITCH': 1,
    'CASE': 1,
    'DEFAULT': 0,
    'LCAL': 2,
    '@DEFINE': 2,
    'IMPORT': 1,
    # Directives (not supported atm)
    'DW': 1
}
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}

lib_root = "Libraries"
default_imports = {"inst.core", "inst.io", "inst.basic", "inst.complex"}


def read_lib(name: str):
    path = lib_root + "/" + name.replace(".", "/") + ".urcl"
    with open(path, "r") as f:
        return f.read()


# ERRORS
class E(Enum):
    illegal_char = "Illegal Char '{}'"
    invalid_char = "Invalid Character {}"
    unk_port = "Unknown port name '{}'"
    miss_pair = "Missing closing quote {}"
    word_miss = "Keyword expected, found {} instead"
    sym_miss = "Symbol '{}' expected"
    tok_miss = "Token expected, found {} instead"
    operand_expected = "Operand expected, found {} instead"
    wrong_op_num = "Instruction {} takes {} operands but got {}"
    invalid_op_type = "Invalid operand type '{}' for Instruction {}"
    wrong_op_type = "Wrong operand type '{}' used"
    duplicate_case = "Duplicate Case '{}' used"
    duplicate_default = "Duplicate Default used"
    duplicate_macro = 'Duplicate macro "{}" used'
    undefined_macro = "Undefined macro '{}' used"
    outside_loop = '{} must be used inside a loop'
    missing_if = '{} must come after "IF" instruction'
    end_expected = 'Missing "END"'
    str = "{}"

    def __repr__(self):
        return self.value


usage = """usage: urclpp <source_file> <destination_file>"""


def main():
    source_name = argv[1] if len(argv) >= 2 else None  
    dest_name = argv[2] if len(argv) >= 3 else None

    source = r'''ADD (LSH R1)[4][]'''
    
    if source_name is not None:
        if isfile(source_name):
            with open(source_name, mode='r') as sf:
                source = sf.read().replace("\r", "")
        else:
            print(f'"{source_name}" is not a file', file=stderr)
            exit(1)

    dest = stdout

    if dest_name is not None:
        dest = open(dest_name, mode="w")

    for lib in default_imports:
        source = read_lib(lib) + "\n" + source

    tokens, lex_errors = Lexer(source).make_tokens()

    print("tokens:", file=dest)
    print(tokens, file=dest)
    print("\n", file=dest)
    
    if len(lex_errors) > 1:
        print(lex_errors, file=stderr)
        exit(1)
    # parse
    parser = Parser(tokens).parse()

    print("Instructions:", file=dest)
    print(parser.instructions, file=dest)
    print("\n", file=dest)

    print("Identifiers:", file=dest)
    print(parser.ids.keys(), file=dest)
    print("\n", file=dest)

    if len(parser.errors) > 1:
        print(parser.errors, file=stderr)
        exit(1)

    return


class Token:
    def __init__(self, type: T, pos: int, line: int, value: str = "") -> None:
        self.type = type
        self.position = pos
        self.line = line
        self.value = value
    
    def __repr__(self) -> str:
        return f"<{self.type} {self.value}>"


class Error:
    def __init__(self, error: E, index: int, line: int, *args) -> None:
        self.error = error
        self.line = line
        self.index = index
        self.args = args

    def __repr__(self) -> str:
        return f'{self.error.value.format(self.args)}, at {self.index} at line {self.line}'


class Lexer:
    def __init__(self, program: str) -> None:
        self.p = program
        self.line_nr = 0
        self.j = 0
        self.i = 0
        self.output: list[Token] = []
        self.errors: list[Error] = []

    def error(self, error: E, extra: str = "") -> None:
        self.errors.append(Error(error, self.j, self.line_nr, extra))

    def token(self, type: T, value: str = "") -> None:
        self.output.append(Token(type, self.j, self.line_nr, value))

    def make_tokens(self):
        while self.has_next():
            while self.has_next() and self.p[self.i] in ' ,\t':  # ignore commas and indentation
                self.advance()
            if self.has_next():
                if self.p[self.i] == '\n':  # change line
                    self.advance()
                    self.new_line()

                elif self.p[self.i] == '/':
                    self.advance()
                    if self.has_next() and self.p[self.i] == '/':  # inline comment
                        self.inline_comment()
                    elif self.has_next() and self.p[self.i] == '*':
                        self.multi_line_comment()
                    else:  # you got your hopes high but it was just an illegal char :/
                        self.error(E.illegal_char, '/')

                elif self.p[self.i] in symbols:
                    self.make_symbol()
                    self.advance()
                else:
                    self.make_operand()

        return self.output, self.errors

    def make_symbol(self):
        if self.p[self.i] == '<':
            if self.has_next() and self.p[self.i] == '=':
                self.token(symbols['<='])
            else:
                self.token(symbols['<'])

        elif self.p[self.i] == '>':
            if self.has_next() and self.p[self.i] == '=':
                self.token(symbols['>='])
            else:
                self.token(symbols['>'])

        elif self.p[self.i] == '[' and self.has_next(-1) and self.p[self.i-1] != ' ':
            self.make_mem_index()

        elif self.p[self.i] == '&' and self.has_next() and self.p[self.i+1] == '&':
            self.advance(2)
            self.token(T.sym_and)

        elif self.p[self.i] == '|' and self.has_next() and self.p[self.i+1] == '|':
            self.advance(2)
            self.token(T.sym_or)

        else:
            self.token(symbols[self.p[self.i]])

    def make_operand(self, indexed: bool = False) -> None:
        if self.p[self.i] in digits + '+-':  # immediate value
            self.token(T.imm, str(self.make_num(indexed)))

        elif self.p[self.i] in charset:  # opcode or other words
            self.token(T.word, self.make_word(indexed))

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.advance()
            if prefix in 'rR$':  # register
                if self.p[self.i] not in '+-':
                    self.token(T.reg, 'R' + str(self.make_num(indexed)))
                else:
                    self.error(E.illegal_char, self.p[self.i])

            elif prefix in 'mM#':  # memory
                if self.p[self.i] not in '+-':
                    self.token(T.mem, 'M' + str(self.make_num(indexed)))
                else:
                    self.error(E.illegal_char, self.p[self.i])

            elif prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.token(T.port, '%' + str(self.make_num()))
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.token(T.port, '%' + name)
                    else:
                        self.error(E.unk_port, name)

            elif prefix == '~':  # relative
                self.token(T.relative, prefix + str(self.make_num(indexed)))

            elif prefix == '.':  # label
                self.token(T.label, prefix + self.make_word(indexed))

            elif prefix == '@':  # macro
                self.token(T.macro, prefix + self.make_word(indexed))

            elif prefix == "'":  # character
                char = self.make_str("'")
                # TODO check invalid char
                # if char == invalid_char:
                #     pass
                if len(char) == 3:  # char = "'<char>'"
                    self.token(T.char, char)
                else:
                    self.error(E.illegal_char, char)

            elif prefix == '"':
                self.token(T.string, self.make_str('"'))

            # elif prefix == '':
            #    self.token()

            else:  # unknown symbol
                if indexed and self.p[self.i] == ']':
                    self.advance()
                else:
                    self.error(E.illegal_char, self.p[self.i-1])

        if self.has_next() and self.p[self.i] == '\n':
            self.new_line()
            self.advance()
            self.token(T.newLine)

    def make_str(self, char: str) -> str:
        word = char
        while self.has_next() and self.p[self.i] != char and self.p[self.i] != '\n':
            word += self.p[self.i]
            self.advance()

        if self.has_next() and self.p[self.i] == '\n':
            self.new_line()
            self.token(T.newLine)
            self.error(E.miss_pair, char)
            # TODO return invalid char
            raise ValueError("invalid char")
            # return invalid_char
        else:
            word += self.p[self.i]
            self.advance()
            return word

    def make_word(self, indexed: bool = False) -> str:
        word = self.p[self.i]
        self.advance()
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] == '[':  # has pointer after the operand
                self.make_mem_index()
                return word
            elif indexed and self.p[self.i] == ']':
                self.advance()
                return word
            if self.p[self.i] in symbols:
                return word
            if self.p[self.i] not in charset:
                self.error(E.illegal_char, self.p[self.i])
            else:
                word += self.p[self.i]
            self.advance()

        return word

    def make_num(self, indexed: bool = False) -> int:
        if self.p[self.i] == ' ':
            return 0
        num = ''
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] == '[':  # has pointer after the operand
                self.make_mem_index()
                return int(num, 0)
            elif indexed and self.p[self.i] == ']':
                self.advance()
                return int(num, 0)

            if self.p[self.i] not in digits + bases:
                self.error(E.illegal_char, self.p[self.i])
            else:
                num += self.p[self.i]
            self.advance()

        return int(num, 0)

    def make_mem_index(self) -> None:
        self.advance()
        while self.p[self.i] == ' ':
            self.advance()
        if self.p[self.i] == ']':
            self.token(T.pointer)
            self.token(T.imm, '0')
            self.advance()
        else:
            self.make_operand(True)     # create and push to output new token
            index = self.output.pop()   # retrieve that token generated to switch orders
            self.token(T.pointer)
            self.output.append(index)

        if self.has_next() and self.p[self.i] == '[':
            self.make_mem_index()

    def multi_line_comment(self) -> None:
        while self.has_next(1) and (self.p[self.i] != '*' or self.p[self.i + 1] != '/'):
            if self.p[self.i] == '\n':
                self.new_line()
            self.advance()
        self.advance()

    def inline_comment(self) -> None:
        while self.has_next() and self.p[self.i] != '\n':
            self.advance()
        self.advance()
        self.new_line()

    def advance(self, i: int = 1):
        self.i += i
        self.j += i

    def new_line(self):
        self.token(T.newLine)
        self.line_nr += 1
        self.j = 0

    def has_next(self, i: int = 0) -> bool:
        return len(self.p) > self.i + i >= 0


@dataclass
class Id:
    pass


class OpPrim(Enum):
    Reg = "reg"
    Imm = "imm"


op_prims = {
    "R": [OpPrim.Reg],
    "I": [OpPrim.Imm],
    "A": [OpPrim.Imm, OpPrim.Reg]
}


@dataclass
class OpDef:
    prims: List[OpPrim]


@dataclass
class InstDef(Id):
    opcode: str
    operands: OrderedDict[str, OpDef]


@dataclass
class LabelDef(Id):
    location: int


@dataclass
class Operand:
    prim: OpPrim
    value: int


class Instruction:
    def __init__(self, opcode: Token, *args: Token) -> None:
        self.opcode = opcode
        self.operands: list[Token] = []
        for i, op in enumerate(args):
            if op is not None:
                self.operands[i] = op

    def __repr__(self) -> str:
        out = f'<INST {self.opcode}'
        for op in self.operands:
            out += ' ' + str(op)
        return out + '>'


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens: list[Token] = tokens
        self.ids: dict[str, Id] = {}
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []
        self.temp: dict[Token] = []
        self.id_count = 0
        self.macros: dict[Token] = {}
        self.labels: set(str) = set()
        self.i = 0

    def error(self, error: E, token: Token, *args):
        self.errors.append(Error(error, token.position, token.line, *args))

    def error_str(self, msg: str):
        # TODO make this make sense lol
        token = self.peak() if self.has_next() else self.tokens[-1]
        self.errors.append(Error(E.str, token.position, token.line, msg))

    def parse(self):
        while self.has_next():
            word = self.next_word()
            if word is None:
                if self.has_next():
                    self.next()
                continue

            if word.value.upper() == 'INST':
                self.make_inst_def()
                continue

            id = self.ids.get(word.value.upper())
            if isinstance(id, InstDef):
                self.make_instruction(id)
                self.i += 1
                continue

            # To prevent infinite loop, we should probably check whether a token was found and report an error instead 
            self.error_str(msg=f"unexpected word {word.value}")
            self.advance()
        return self

    def make_inst_def(self) -> None:
        self.advance()
        name = self.next_word()
        if name is None:
            self.error(E.tok_miss, self.tokens[-1])
            return

        inst = InstDef(opcode=name.value.upper(), operands=OrderedDict())
        if self.ids.get(name.value) is not None:
            self.error_str(f"identifier {name} is already defined")
            return
        self.ids[name.value] = inst

        self.advance()
        while self.has_next() and self.tokens[self.i].type is not T.newLine:
            op_name = self.next()
            if op_name is None:
                self.error_str(msg="missing operant name")
                self.skip_line()
                return
            if not self.has_next() or self.next().type is not T.sym_col:
                self.error_str(msg="missing colon")
                self.skip_line()
                return

            prim_str = self.next()
            prim = None if prim_str is None else op_prims.get(prim_str.value)
            if prim is None:
                self.error_str(msg="missing type (R, I or A)")
                return
            inst.operands[op_name.value] = OpDef(prims=prim)

        pass

    # loop saves the context of the nearest outer loop
    # loop[0] = final statement for skip
    # loop[1] = start_label
    # loop[2] = end_label
    def make_instruction(self, recursive: bool = False, loop=None) -> None:
        inst = Instruction(self.tokens[self.i])
        operands = self.make_operands(inst, recursive=recursive)
        for operand, (name, op) in zip(operands, instdef.operands.items()):
            if operand.prim not in op.prims:
                self.error_str(msg=f"Operant {name} of Instruction {instdef.opcode} must be one of {op.prims}")

        if len(operands) != len(instdef.operands):
            self.error_str(
                msg=f"Instruction {instdef.opcode} takes {len(instdef.operands)} operands but got {len(operands)}")

        # # # # # # # # # # # # not final # # # # # # # # # # # #

        if self.peak().type == T.word:
            if self.peak().value.upper() == 'EXIT':
                if loop is None:
                    self.error(E.outside_loop, self.peak(), self.peak())
                else:
                    self.add_inst(Instruction(Token(T.word, -1, -1, 'BRA'), loop[2]))
                self.skip_line('EXIT', True)

            elif self.peak().value.upper() == 'SKIP':
                if loop is None:
                    self.error(E.outside_loop, self.peak(), self.peak())
                else:
                    if loop[0] is not None:
                        self.add_inst(end_statement)
                    self.add_inst(Instruction(Token(T.word, -1, -1, 'BRA'), loop[1]))
                self.skip_line('SKIP', True)

            elif self.peak().value.upper() == 'SWITCH':
                self.make_switch()

            elif self.peak().value.upper() == 'IF':
                self.make_if()

            elif self.peak().value.upper() == 'FOR':
                self.make_for()

            elif self.peak().value.upper() == 'WHILE':
                self.make_while()

            elif self.peak().value.upper() == 'DEFINE':
                self.make_define()

            elif self.peak().value.upper() == 'LCAL':
                self.make_lcal()

            # add more later

        inst.operands = operands
        self.instructions.append(inst)

    def make_operands(self, inst: Instruction, recursive: bool = False) -> List[Operand]:
        operands: List[Operand] = []
        while self.tokens != T.newLine:
            operand = self.next_operand(inst)
            if operand is not None:
                operands.append(operand)

        return operands

    def process_scope(self, recursive=False, final_inst=None, start_label=None, end_label=None):
        while self.has_next() and (self.tokens.type != T.word or self.tokens.value.upper() != 'END'):
            params: list[Instruction, Token, Token] = list(final_inst, start_label, end_label)
            self.make_instruction(recursive, params)
        if not self.has_next():
            self.error(E.end_expected, self.peak())

    def peak(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        self.i += 1
        return self.tokens[self.i-1]

    def skip_line(self, inst: str = None, error: bool = False) -> int:
        skipped = self.skip_until(T.newLine)
        if error and skipped != 0:
            self.error_str(msg=f"Instruction {inst} takes {0} operands but got {n}")
        if self.has_next():
            self.advance()
        return

    def skip_until(self, type: T) -> int:
        skipped = 0
        while self.has_next() and self.next().type is not type:
            skipped += 1
        return skipped

    def add_inst(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def add_tmp(self, tmp: Token) -> None:
        if tmp is not None and tmp.type == T.reg:
            self.temp[Token] = True

    def get_tmp(self):
        for tmp in self.temp:
            if self.temp[tmp]:
                self.temp[tmp] = false
                return tmp
        return  # not enough temp registers declared

    def ret_tmp(self, tmp: Token):
        self.temp[tmp] = True

    def get_opcode(self):
        while self.has_next() and self.tokens[self.i].type != T.word:
            self.error(E.word_miss, self.tokens[self.i], str(self.tokens[self.i]))
            self.advance()
        if self.has_next():
            return self.tokens[self.i]
        else:
            self.error(E.word_miss, self.tokens[self.i-1], 'nothing')
            return

    def next_operand(self, inst: Instruction) -> Token:
        if self.has_next() and self.peak() != T.newLine:
            type = self.peak().type
            value = self.peak().value

            if type == T.sym_lpa:
                self.advance()
                self.make_instruction(recursive=True)

            elif type == T.macro:
                if value in self.macros:
                    return self.macros[value]
                else:
                    if inst.opcode.upper() != 'DEFINE' or 'DEFINE' != self.tokens[self.i - 1].value: # checking if not the 1st op of DEFINE
                        self.error(E.undefined_macro, self.peak(), self.peak())

            if self.tokens[self.i].type == T.pointer:
                self.advance()
                '''if self.has_next() and self.tokens[self.i] != T.newLine:  # needs tweaking if we want to use tmp regs
                    if len(inst.operands) > 0 and inst.operands[0] is not None:
                        add = Instruction(
                            Token(T.word, None, None, 'ADD'), inst.operands[0], inst.operands[0], self.tokens[self.i])
                        lod = Instruction(Token(T.word, None, None, 'ADD'), inst.operands[0], inst.operands[0])
                        self.instructions.append(add)
                        self.instructions.append(lod)
                        return inst.operands[0]  # repeating the tmp operand

                    else:  # TODO: decode inst to save on a tmp reg, and load that reg to op1
                        pass'''

        return False

    def make_switch(self) -> None:
        inst = Instruction(Token(T.word, self.peak().position, self.peak().line, 'SWITCH'))
        self.advance()

        has_default = False
        default_label: str = None
        end_label, start_label, id = self.get_label_helper(inst)
        case_num = 0
        cases: set(int) = set()
        switch: dict[int] = {}
        dw_loc = len(self.instructions)

        reg: Token = self.next_operand(inst)
        tmp = self.get_tmp()
        self.ret_tmp(tmp)

        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            if self.peak().type == T.word and self.peak().value.upper() == 'CASE':
                operands: List[Token] = self.make_operands(Instruction(Token(T.word, -1, -1, 'SWITCH')))
                case_label = f'.reserved_case{id}_{case_num}'
                self.labels.add(case_label)

                for op in operands:
                    if op.type == T.imm:
                        value = int(op.value, 0)
                        if value in cases:
                            self.error(E.duplicate_case, op)
                        else:
                            cases.add(value)
                            switch[value] = case_label
                            self.add_inst(Instruction(Token(T.label, -1, -1, case_label)))
                    else:
                        self.error(E.invalid_op_type, op, op.type, inst)

            elif self.peak().type == T.word and self.peak().value.upper() == 'DEFAULT':
                if has_default:
                    self.error(E.duplicate_default, self.peak())
                else:
                    has_default = True
                    default_label = f'.reserved_default{id}'
                    self.labels.add(default_label)
                    self.add_inst(Instruction(Token(T.label, -1, -1, default_label)))
                    self.skip_line('DEFAULT', True)

            elif self.peak().type == T.word and self.peak().value.upper() == 'EXIT':
                self.skip_line('EXIT', True)
                self.add_inst(Instruction(Token(T.word, pos, line, 'BRA'), end_label))

            else:
                self.make_instruction()

        if len(cases) > 0:
            biggest_case = cases[0]
            smallest_case = cases[0]
            for case in cases:
                if case > biggest_case:
                    biggest_case = case

                elif case < smallest_case:
                    smallest_case = case

            dw = []
            for num in range(smallest_case, biggest_case + 1):
                if num in cases:
                    address = switch[num]
                else:
                    if default_done:
                        address = default_label
                    else:
                        address = end_label
                dw.append(address)

            dw = str(dw).replace(',', '')
            dw = dw.replace("'", '')
            # inserted in reverse order to what they are here
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'DW'), dw))
            self.instructions.insert(dw_loc, Instruction(Token(T.label, -1, -1, start_label)))
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'JMP'), tmp))
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'LOD'), tmp, tmp))
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'ADD'), tmp, tmp, start_label))
            if smallest_case != 0:
                self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'SUB'), tmp, reg, smallest_case))
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'BRG'), end_label, reg, biggest_case))
            self.instructions.insert(dw_loc, Instruction(Token(T.word, -1, -1, 'BRL'), end_label, reg, smallest_case))

        else:
            self.error(E.word_miss, self.peak(), self.peak())

        self.advance()
        return

    def make_if(self) -> None:
        inst = Instruction(Token(T.word, self.peak().position, self.peak().line, 'IF'))
        self.advance()
        else_done = False
        else_count = 0
        end_label, start_label, id = self.get_label_helper(inst)  # end_label, start_label are not used
        next_label: Token = Token(T.label, pos, line, f'.reserved_else{id}_{else_count}')

        self.do_condition(inst, next_label, id)

        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            if self.peak().type == T.word and self.peak().value.upper() == 'ELIF':
                if else_done:
                    self.error(E.missing_if, self.peak(), self.peak())
                else:
                    self.add_inst(Instruction(Token(T.word, -1, -1, 'JMP'), end_label))
                    self.labels.add(next_label)
                    self.add_inst(Instruction(Token(T.label, -1, -1, next_label)))
                    self.advance()
                    else_count += 1
                    next_label = Token(T.label, pos, line, f'.reserved_else{id}_{else_count}')
                    self.do_condition(inst, next_label, id)

            elif self.peak().type == T.word and self.peak().value.upper() == 'ELSE':
                if else_done:
                    self.error(E.missing_if, self.peak(), self.peak())
                else:
                    self.add_inst(Instruction(Token(T.word, -1, -1, 'JMP'), end_label))
                    self.labels.add(next_label)
                    self.add_inst(Instruction(Token(T.label, -1, -1, next_label)))
                    self.skip_line(inst, True)
            else:
                self.make_instruction()

        if not else_done:
            self.labels.add(next_label)
            self.add_inst(Instruction(Token(T.label, -1, -1, next_label)))
        self.labels.add(end_label)
        self.add_inst(Instruction(Token(T.label, -1, -1, end_label)))

        self.advance()
        return

    def make_for(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position, self.peak().line, 'FOR'))
        self.advance()
        end_label, start_label, id = self.get_label_helper(inst)
        reg = self.next_operand(inst)
        end_num = self.next_operand(inst)

        self.labels.add(start_label)
        self.add_inst(Instruction(Token(T.label, -1, -1, start_label)))
        self.add_inst(Instruction(Token(T.word, -1, -1, 'BRG'), end_label, reg, end_num))

        if self.has_next() and self.tokens[self.i] != T.newLine:  # checking if there is the option parameter
            value_to_add = self.next_operand(inst)
            end_statement: Instruction = Instruction(Token(T.word, -1, -1, 'ADD'), reg, reg, value_to_add)
        else:
            end_statement: Instruction = Instruction(Token(T.word, -1, -1, 'ADD'), reg, reg, Token(T.imm, -1, -1, '1'))
        self.skip_line(inst, True)

        self.process_scope(recursive, end_statement, start_label, end_label)

        if self.has_next():
            self.add_inst(end_statement)
            self.add_inst(Instruction(Token(T.word, -1, -1, 'BRA'), start_label))
            self.labels.add(end_label)
            self.add_inst(Instruction(Token(T.label, -1, -1, end_label)))
        return

    def make_while(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position, self.peak().line, 'WHILE'))
        self.advance()
        end_label, start_label, id = self.get_label_helper(inst)
        self.labels.add(start_label)
        self.add_inst(Instruction(Token(T.label, -1, -1, start_label)))

        self.do_condition(inst, end_label, id)

        self.process_scope(recursive, None, start_label, end_label)

        if self.has_next():
            self.add_inst(Instruction(Token(T.word, -1, -1, 'BRA'), start_label))
            self.labels.add(end_label)
            self.add_inst(Instruction(Token(T.label, -1, -1, end_label)))
        return

    def make_lcal(self) -> None:
        self.advance()

        return

    def make_define(self) -> None:
        inst = Instruction(Token(T.word, self.peak().position, self.peak().line, 'DEFINE'))
        self.advance()
        macro = self.next_operand(inst)
        if macro.type != T.macro:
            self.error(E.invalid_op_type, macro, macro.type, 'DEFINE')

        value = self.next_operand(inst)
        if macro in self.macros:
            self.error(E.duplicate_macro, macro, macro)
        else:
            self.macros[macro] = value

        self.skip_line('DEFINE', True)
        return

    def get_label_helper(self, inst: Instruction) -> Token:
        end_label = Token(T.label, pos, line, f'.reserved_end{self.id_count}')
        loop_label = Token(T.label, pos, line, f'.reserved_{inst}{self.id_count}')
        self.id_count += 1
        return end_label, loop_label, self.id_count-1

    def do_condition(self, inst: Instruction, label: Token, id):
        operands = self.make_operands(inst)
        if len(operands) == 1:
            self.add_inst(Instruction(Token(T.word, -1, -1, 'BRZ'), label, operands[0]))
        elif len(operands) == 3:
            cnd = operands[1].type
            if operands[1] == '':
                pass

        self.add_inst(Instruction(Token(T.label, -1, -1, f'.reserved_body{id}')))
        return

    def shunting_yard(self, inst: Instruction) -> List[Token]:
        queue: List[Token] = []
        stack = []
        while self.has_next() and self.peak().type != T.newLine:
            type = self.peak().type
            if type in op_precedence:
                if op_precedence[type] > op_precedence[stack[-1].type]:
                    stack.append(self.peak())
                else:
                    while op_precedence[type] > op_precedence[stack[-1].type]:
                        queue.append(stack[-1])
                    stack.append(self.peak())
            elif type == T.sym_rpa:
                while len(stack) > 0 and stack[-1].type != T.sym_lpa:
                    queue.append(stack[-1])
                if len(stack) > 0:
                    stack.pop()
                else:
                    self.error(E.word_miss, self.peak(), 'nothing')
            else:
                queue.append(self.next_operand(inst))
            self.advance()

        if self.has_next():
            while len(stack) > 0:
                if stack[-1].type == T.sym_lpa:
                    self.error(E.sym_miss, ')')
                else:
                    queue.append(stack[-1])
        else:
            self.error(E.word_miss, self.peak(), 'nothing')
        self.skip_line()
        return queue

    def next_word(self):
        while self.has_next() and self.tokens[self.i].type != T.word:
            if self.tokens[self.i].type == T.newLine:
                return  # operands only and not opcode error, ignore this line and proceed
            # Opcode Expected, found operand error
            self.i += 1

        if self.has_next():
            return self.tokens[self.i]
        else:  # missing Instruction error
            return

    def advance(self):
        if self.has_next():
            self.i += 1
        else:
            self.error(E.tok_miss, self.tokens[self.i], 'Nothing')

    def has_next(self, i: int = 0):
        return self.i + i < len(self.tokens)


if __name__ == "__main__":
    main()
