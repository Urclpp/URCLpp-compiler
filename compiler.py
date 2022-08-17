import os
from sys import argv, stdout, stderr
from enum import Enum
from typing import List, Tuple, Dict
from math import frexp

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
    char = 'opr_cha'
    string = 'opr_str'
    array = 'opr_arr'
    sym_lpa = 'sym_lpa'
    sym_rpa = 'sym_rpa'
    sym_lbr = 'sym_lbr'
    sym_rbr = 'sym_rbr'
    sym_col = "sym_col"
    sym_gt = "sym_gt"
    sym_lt = "sym_lt"
    sym_geq = "sym_geq"
    sym_leq = "sum_leq"
    sym_equ = "sym_equ"
    sym_dif = "sym_dif"
    sym_and = "sym_and"
    sym_or = "sym_or"
    sym_lnbr = 'sym_lnbr'
    group = 'group'

    def __repr__(self) -> str:
        return self.value


# CONSTANTS
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890.'
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
    ';': T.sym_lnbr,
}
op_precedence = {
    T.sym_lt: 0,
    T.sym_gt: 0,
    T.sym_geq: 0,
    T.sym_leq: 0,
    T.sym_dif: 1,
    T.sym_equ: 1,
    T.sym_and: 2,
    T.sym_or: 3,
    T.sym_lpa: 5
}
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}
default_macros = {'@BITS', '@MINREG', '@MINRAM', '@HEAP', '@MINSTACK', '@MSB', '@SMSB', '@MAX', '@SMAX', '@UHALF',
                  '@LHALF'}

file_extension = '.urcl'
file_extensionpp = '.urclpp'
compiler_root = os.path.dirname(__file__)
lib_root = os.path.join(compiler_root, 'urclpp-libraries')
default_imports = {"inst.core", "inst.io", "inst.basic", "inst.complex"}


# ERRORS
class E(Enum):
    illegal_char = "Illegal Char '{}'"
    invalid_char = "Invalid Character {}"
    invalid_literal = "Invalid literal for imm value {}, must be either 16 32 or 64"
    unk_port = "Unknown port name '{}'"
    unk_library = "Unknown library name '{}'"
    unk_function = "Unknown library function '{}'"
    unk_instruction = "Unknown instruction name '{}'"
    miss_pair = "Missing closing quote {}"
    word_miss = "Keyword expected, found {} instead"
    sym_miss = "Symbol '{}' expected"
    tok_miss = "Token expected, found {} instead"
    operand_expected = "Operand expected, found {} instead"
    wrong_op_num = "Instruction {} takes {} operands but got {}"
    invalid_op_type = "Invalid operand type '{}' for Instruction {}"
    wrong_op_type = "Wrong operand type '{}' used, '{}' expected"
    duplicate_case = "Duplicate Case '{}' used"
    duplicate_default = "Duplicate Default used"
    duplicate_macro = 'Duplicate macro "{}" used'
    duplicate_label = 'Duplicate label "{}" used'
    undefined_macro = "Undefined macro '{}' used"
    undefined_label = "Undefined label '{}' used"
    outside_loop = '{} must be used inside a loop'
    missing_if = '{} must come after "IF" instruction'
    end_expected = 'Missing "END"'
    no_tmp = "Not enough temporary registers defined"
    str = "{}"

    def __repr__(self):
        return self.value


usage = """usage: urclpp <source_file> <destination_file>"""


def main():
    source_name = argv[1] if len(argv) >= 2 else None
    dest_name = argv[2] if len(argv) >= 3 else None

    if source_name == '--help':
        print(usage)
        return

    source = r'''IF 1<2 && r1'''

    output_file_name = dest_name
    label_id = f'.reserved_{output_file_name}_'

    if source_name is not None:
        if os.path.isfile(source_name):
            with open(source_name, mode='r') as sf:
                source = sf.read().replace("\r", "")
        else:
            print(f'"{source_name}" is not a file', file=stderr)
            exit(1)

    dest = stdout

    if dest_name is not None:
        dest = open(dest_name, mode="w")

    tokens, lex_errors = Lexer(source, label_id, dest_name).make_tokens()

    # print("tokens:", file=dest)
    # print(tokens, file=dest)
    # print("\n", file=dest)

    if len(lex_errors) > 0:
        print(lex_errors, file=stderr)
        exit(1)
    # parse
    parser = Parser(tokens, label_id, dest_name)

    for lib in default_imports:  # generating all the default inst_def needed to execute the code
        parser.read_lib(lib)

    parser.parse()

    # print("Instructions:", file=dest)
    for inst in parser.instructions:
        print(str(inst), file=dest)
    # print(parser.instructions, file=dest)
    print("\n", file=dest)

    # print("Identifiers:", file=dest)
    # print(parser.ids.keys(), file=dest)
    # print("\n", file=dest)

    if len(parser.errors) > 0:
        for err in parser.errors:
            print(err, file=stderr)
        exit(1)

    return


class Token:
    def __init__(self, type: T, pos: int, line: int, value: str = "") -> None:
        self.type: T = type
        self.position = pos
        self.line = line
        self.value = value

    def __repr__(self) -> str:
        return f"<{self.type} {self.value}>"


class Error:
    def __init__(self, error: E, index: int, line: int, file_name: str, *args) -> None:
        self.error = error
        self.line = line
        self.index = index
        self.args = args
        self.file_name = file_name

    def __repr__(self) -> str:
        return f'{self.file_name}:{self.index}:{self.line}: {self.error.value.format(*self.args)}'


class Lexer:
    def __init__(self, program: str, label_id: str, file_name: str) -> None:
        self.p = program + '\n'     # newline to avoid problems
        self.line_nr = 0
        self.j = 0
        self.i = 0
        self.output: List[Token] = []
        self.errors: List[Error] = []
        self.label_id = label_id
        self.file_name = file_name

    def error(self, error: E, extra: str = "") -> None:
        self.errors.append(Error(error, self.j, self.line_nr+1, self.file_name, extra))

    def token(self, type: T, value: str = "") -> None:
        self.output.append(Token(type, self.j, self.line_nr+1, value))

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
                    else:  # you got your hopes high, but it was just an illegal char :/
                        self.error(E.illegal_char, '/')

                else:
                    self.make_operand()

        return self.output, self.errors

    def make_operand(self) -> None:
        if self.p[self.i] == '[':
            values = self.make_array()
            self.token(T.array, values)
            self.advance()

        elif self.p[self.i] in symbols or self.p[self.i] in {'=', '!', '&', '|'}:  # gotta check for first char of all
            self.make_symbol()
            self.advance()

        elif self.p[self.i] in digits + '+-':  # immediate value
            self.token(T.imm, self.make_number())

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.advance()
            if not self.has_next() or self.p[self.i] in indentation:
                if prefix in charset:
                    self.token(T.word, prefix)
                else:
                    self.error(E.illegal_char, self.p[self.i - 1])
                return

            if prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.token(T.port, '%' + str(self.make_int()))
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.token(T.port, '%' + name)
                    else:
                        self.error(E.unk_port, name)

            elif prefix == '~':  # relative
                self.token(T.relative, prefix + str(self.make_int()))

            elif prefix == '.':  # label
                self.token(T.label, self.label_id + self.make_word())

            elif prefix == '@':  # macro
                self.token(T.macro, prefix + self.make_word())

            elif prefix == "'":  # character
                char = self.make_str("'")
                if len(char) == 3:  # char = "'<char>'"
                    self.token(T.char, char)
                else:
                    self.error(E.invalid_char, char)

            elif prefix == '"':
                self.token(T.string, self.make_str('"'))

            elif self.p[self.i] in charset:  # opcode or other words
                string = prefix + self.make_word()
                if string.upper() in {'SP', 'PC'}:
                    self.token(T.reg, string)
                elif prefix in 'mM#rR$':
                    try:
                        if prefix in 'rR$':  # register
                            self.token(T.reg, 'R' + str(int(string[1:], 0)))
                        else:  # memory
                            self.token(T.mem, 'M' + str(int(string[1:], 0)))
                    except ValueError:
                        self.token(T.word, string)
                else:
                    self.token(T.word, string)

                if self.has_next() and self.p[self.i] == '[':
                    self.token(T.sym_lbr)
                    self.advance()

            # elif prefix == '':
            #    self.token()

            else:  # unknown symbol
                self.error(E.illegal_char, self.p[self.i - 1])

        if self.has_next() and self.p[self.i] == '\n':
            self.new_line()
            self.advance()

    def make_symbol(self):
        if self.p[self.i] == '<':
            if self.has_next(1) and self.p[self.i + 1] == '=':
                self.token(symbols['<='])
                self.advance()
            else:
                self.token(symbols['<'])

        elif self.p[self.i] == '>':
            if self.has_next(1) and self.p[self.i + 1] == '=':
                self.token(symbols['>='])
                self.advance()
            else:
                self.token(symbols['>'])

        elif self.p[self.i] == '=':
            if self.has_next(1) and self.p[self.i] == '=':
                self.token(T.sym_equ)
                self.advance()
            else:
                self.error(E.illegal_char, self.p[self.i])

        elif self.p[self.i] == '!':
            if self.has_next(1) and self.p[self.i] == '=':
                self.token(T.sym_dif)
                self.advance()
            else:
                self.error(E.illegal_char, self.p[self.i])

        elif self.p[self.i] == '&' and self.has_next(1) and self.p[self.i + 1] == '&':
            self.advance()
            self.token(T.sym_and)

        elif self.p[self.i] == '|' and self.has_next(1) and self.p[self.i + 1] == '|':
            self.advance()
            self.token(T.sym_or)

        elif self.p[self.i] == ';':
            self.token(T.newLine)

        else:
            try:
                self.token(symbols[self.p[self.i]])
            except KeyError:
                self.error(E.illegal_char, self.p[self.i])

    def make_number(self):   # can create errors if imm is the last thing on file
        if self.p[self.i] in '+-':  # detecting sign
            num = self.p[self.i]
            self.advance()
        else:
            num = ''
        base = 10
        if self.has_next(1) and self.p[self.i] == '0':   # detecting the base
            num += '0'
            self.advance()
            if self.p[self.i].lower() == 'x':
                base = 16
            elif self.p[self.i].lower() == 'd':
                base = 10
            elif self.p[self.i].lower() == 'o':
                base = 8
            elif self.p[self.i].lower() == 'b':
                base = 2
            else:
                self.advance(-1)
            self.advance()
        integer = 0
        decimals = ''
        exponent = ''
        has_point = False
        has_exponent = False
        while self.has_next() and self.p[self.i] in digits + '.fe':
            if has_point:
                if self.p[self.i] == '.':
                    self.error(E.illegal_char, self.p[self.i])
                    break
                elif self.p[self.i] == 'e':
                    if has_exponent:
                        self.error(E.illegal_char, self.p[self.i])
                        break
                    else:
                        has_exponent = True
                elif self.p[self.i] == 'f':
                    self.advance()
                    if exponent == '':
                        exponent = 0
                    else:
                        exponent = int(exponent, base)
                    decimals = int(decimals, base) / (base ** len(decimals))
                    integer += decimals
                    integer *= (base ** exponent)
                    if self.p[self.i] == 'x':  # its a fixed point
                        self.advance()
                        offset = self.make_int(base)
                        integer *= offset
                        integer = int(integer)
                        return integer
                    else:  # its a float
                        if self.p[self.i] in indentation:
                            bits = 32
                        else:
                            bits = self.make_int()
                            if bits not in {16, 32, 64}:
                                self.error(E.invalid_literal, bits)
                        return self.make_float(integer, bits)
                elif has_exponent:
                    exponent += self.p[self.i]
                else:
                    decimals += self.p[self.i]
            elif self.p[self.i] == '.':
                has_point = True
                integer = int(num, base)
            elif self.p[self.i] == 'f':
                self.advance()
                if decimals == '':
                    decimals = 0
                else:
                    decimals = int(decimals, base) / (base ** len(decimals))
                if exponent == '':
                    exponent = 0
                else:
                    exponent = int(exponent, base)
                integer = int(num, base)
                integer += decimals
                integer *= (base ** exponent)
                if self.p[self.i] == 'x':   # its a fixed point
                    self.advance()
                    offset = self.make_int(base)
                    integer *= offset
                    integer = int(integer)
                    return integer
                else:   # its a float
                    if self.p[self.i] in indentation:
                        bits = 32
                    else:
                        bits = self.make_int()
                        if bits not in {16, 32, 64}:
                            self.error(E.invalid_literal, bits)
                    return self.make_float(integer, bits)

            elif self.p[self.i] == 'e':
                if has_exponent:
                    self.error(E.illegal_char, self.p[self.i])
                    break
                else:
                    has_exponent = True
            elif has_exponent:
                exponent += self.p[self.i]
            else:
                num += self.p[self.i]
            self.advance()
        if has_point:
            if has_exponent:
                return int(num, base) * (base ** int(exponent, base))
            else:
                return int(num, base)
        else:
            if has_exponent:
                return int(num, base) * (base ** int(exponent, base))
            else:
                return int(num, base)

    def make_float(self, num, bits) -> int:
        if num < 0:
            output = 2 ** (bits-1)
        else:
            output = 0
        mantissa, exponent = frexp(num)
        mantissa -= 0.5
        exponent -= 1
        if bits == 16:
            exponent += 15
            mantissa *= 2 << 10
            exponent *= 2 << 11
            pass
        elif bits == 32:
            exponent += 127
            mantissa *= 2 << 23
            exponent *= 2 << 22
            pass
        elif bits == 64:
            exponent += 1023
            mantissa *= 2 << 51
            exponent *= 2 << 52

        output += int(mantissa)
        output += exponent
        return output

    def make_str(self, char: str) -> str:
        word = char
        while self.has_next() and self.p[self.i] != char and self.p[self.i] != '\n':
            word += self.p[self.i]
            self.advance()

        if self.has_next() and self.p[self.i] == '\n':
            self.new_line()
            self.token(T.newLine)
            self.error(E.miss_pair, char)
            # raise ValueError("invalid char")
            return word + "'"
        else:
            word += self.p[self.i]
            self.advance()
            return word

    def make_word(self) -> str:
        word = ''
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] in symbols:
                break
            elif self.p[self.i] not in charset:
                self.error(E.illegal_char, self.p[self.i])
            else:
                word += self.p[self.i]
            self.advance()
        return word

    def make_array(self):
        self.advance()
        values = []

        while self.has_next() and self.p[self.i] != ']':
            while self.has_next() and self.p[self.i] in indentation:
                self.advance()
            if self.has_next():
                if self.p[self.i] == ']':
                    break
                self.make_operand()
                values.append(self.output.pop())
            else:
                self.error(E.sym_miss, ']')
                break

        if len(values) == 0:
            self.error(E.operand_expected, 'nothing')
            return None
        else:
            return values

    def make_int(self, base=0) -> int:
        if self.p[self.i] == ' ':
            return 0
        if self.p[self.i] in '+-':
            num = self.p[self.i]
            self.advance()
        else:
            num = ''
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] in symbols:
                break
            elif self.p[self.i] not in digits + bases:
                self.error(E.illegal_char, self.p[self.i])
            else:
                num += self.p[self.i]
            self.advance()
        return int(num, base)

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
        self.j = 1

    def has_next(self, i: int = 0) -> bool:
        return len(self.p) > self.i + i >= 0


@dataclass
class OpType:
    def __init__(self):
        self.op_type = {'REG', 'IMM', 'WB', 'MEM', 'LOC', 'ANY', 'IO'}
        self.operand1_type = {  # used for the first operand
            'LOC': {T.reg, T.label, T.imm, T.relative},
            'MEM': {T.reg, T.label, T.imm, T.mem},
            'WB': {T.reg},
            'IO': {T.port},
            'ANY': {T.reg, T.imm, T.mem, T.label, T.char, T.relative}
        }
        self.operand_type = {
            'IO': {T.port},
            'REG': {T.reg},
            'MEM': {T.reg, T.label, T.imm, T.mem},
            'LOC': {T.reg, T.label, T.imm, T.relative},
            'IMM': {T.imm, T.mem, T.label, T.char, T.relative},
            'ANY': {T.reg, T.imm, T.mem, T.label, T.char, T.relative}
        }

    def allowed_types(self, type, op1: bool = False):
        if op1:
            if type in self.operand1_type:
                return self.operand1_type[type]
        else:
            if type in self.operand_type:
                return self.operand_type[type]
        return


ot = OpType()


class InstDef:
    def __init__(self, opcode: str, *args: str) -> None:
        self.opcode = opcode
        self.operands: List[str] = []
        for op in args:
            self.operands.append(op)  # pre: check dont pass a None or an invalid key. need to check b4

    def __repr__(self) -> str:
        out = f'<INSTDEF {self.opcode}'
        for op in self.operands:
            out += ' ' + str(op)
        return out + '>'


class Instruction:
    def __init__(self, opcode: Token, inst_def, *args: Token) -> None:
        self.opcode = opcode
        self.operands: List[Token] = []
        self.definition: InstDef = inst_def
        self.post_inst = None
        for op in args:
            if op is not None:
                self.operands.append(op)

    def add_inst_later(self, inst):  # adds an instruction right after the current one is added
        self.post_inst = inst
        return

    def __str__(self):
        string = self.opcode.value
        for op in self.operands:
            if op.type == T.imm:
                string += ' ' + str(op.value)

            elif op.type == T.array:
                string += " ["
                for val in op.value:
                    string += f' {val.value}'

                string += ' ]'

            elif op.type == T.string:
                s = list(op.value[1:-1])
                s.insert(0, len(op.value) - 2)
                string += ' ' + str(s).replace(',', '')

            else:
                string += ' ' + op.value
        return string

    def __repr__(self) -> str:
        out = f'<INST {self.opcode}'
        for op in self.operands:
            out += ' ' + str(op)
        return out + '>'


def token(type: T, value=''):
    return Token(type, -1, -1, value)


class Parser:
    def __init__(self, tokens: List[Token], label_id: str, file_name: str, recursive: bool = False):
        self.file_name = file_name
        self.tokens: List[Token] = tokens
        self.instructions: List[Instruction] = []
        self.inst_def: Dict[str, InstDef] = {}
        self.errors: List[Error] = []

        self.temp: Dict[Token] = {token(T.reg, 'tmp1'): True, token(T.reg, 'tmp2'): True}
        self.id_count = 0
        self.macros: Dict[Token] = {}
        self.labels = set()
        self.label_id = label_id

        self.lib_headers: Dict[str] = {}
        self.imported_libs = set()
        self.recursive = recursive
        if recursive:
            self.lib_code = []
        else:
            self.lib_code = [Instruction(token(T.word, 'HLT'), None)]
        self.i = 0

    def error(self, error: E, tok: Token, *args):
        self.errors.append(Error(error, tok.position, tok.line, self.file_name, *args))

    def add_inst(self, inst: Instruction) -> None:
        self.instructions.append(inst)
        if inst.post_inst is not None:
            self.add_inst(inst.post_inst)

    def make_inst_def(self):
        opcode = self.get_opcode()
        inst_def = InstDef(opcode.value.upper())
        self.advance()
        while self.has_next() and self.peak().type != T.newLine:
            if self.peak().type == T.word and self.peak().value.upper() in ot.op_type:
                inst_def.operands.append(self.peak().value.upper())
            else:
                self.error(E.wrong_op_type, self.peak(), self.peak(), 'OP_def_type')
            self.advance()

        if opcode not in self.inst_def:
            self.inst_def[opcode.value.upper()] = inst_def
        else:  # already exists a definition
            pass
        return

    def get_lib_headers(self):
        headers = {
            'BITS': None,
            'OUTS': None,
            'OPS': None,
            'REG': None
        }
        while self.has_next():
            header = self.next_word()
            if header is not None and header.value in headers:
                self.advance()
                inst = Instruction(header, None)
                if inst.opcode.value in {'BITS'}:
                    comparison_op = self.next_operand(inst)
                op = self.next_operand(inst)
                if op is not None and op.type == T.imm:
                    headers[header.value] = op.value
            else:  # headers are over
                if self.i != 0:
                    self.i -= 1
                else:
                    pass  # should get error cause no headers were provided
                break
            self.skip_line()
        return headers

    def parse(self):
        while self.has_next():
            while self.has_next() and self.peak().type == T.newLine:
                self.advance()
            if not self.has_next():
                break
            tok = self.peak()
            if tok.type == T.label and tok.value in self.labels:
                self.error(E.duplicate_label, tok, tok.value)
                self.tokens.remove(tok)
            else:
                self.labels.add(tok.value)

            self.skip_line()

        self.i = 0
        while self.has_next():
            while self.has_next() and self.peak().type == T.newLine:
                self.advance()
            if not self.has_next():
                break

            tok = self.peak()
            if tok.type == T.label:
                self.add_inst(Instruction(tok, None))
                self.advance()

            elif tok.type == T.word:
                self.make_instruction()

            else:
                self.error(E.word_miss, tok, self.peak())
                self.advance()

        if not self.recursive:
            self.instructions = self.instructions + self.lib_code
        return self

    # loop saves the context of the nearest outer loop
    # loop[0] = final statement for skip
    # loop[1] = start_label
    # loop[2] = end_label
    def make_instruction(self, recursive: bool = False, loop=None):
        opcode = self.get_opcode()
        opcode.value = opcode.value.upper()
        self.advance()
        if not self.has_next():     # useless but might save from some exceptions
            return

        temps: Dict[Token] = self.temp.copy()  # save the current state of the temps

        opcode_str = opcode.value
        if recursive:
            tmp = self.get_tmp()
            if opcode_str == 'LCAL':
                self.make_lcal()
                self.add_inst(Instruction(token(T.word, 'MOV'), None, tmp, token(T.reg, 'R1')))
            else:
                inst = Instruction(opcode, self.get_inst_def(opcode))
                inst.operands.append(tmp)
                self.make_operands(inst, recursive=recursive)
                self.instructions.append(inst)
            return tmp

        if opcode_str == 'INST':
            self.make_inst_def()

        elif opcode_str == 'EXIT':
            if loop is None:
                self.error(E.outside_loop, self.peak(-1), self.peak(-1))
            else:
                self.add_inst(Instruction(token(T.word, 'JMP'), None, loop[2]))

        elif opcode_str == 'SKIP':
            if loop is None:
                self.error(E.outside_loop, self.peak(-1), self.peak(-1))
            else:
                if loop[0] is not None:  # while loops dont have a final statement
                    self.add_inst(loop[0])
                self.add_inst(Instruction(token(T.word, 'JMP'), None, loop[1]))

        elif opcode_str == 'END':  # ends are recognized by the funcs that use them, so we will ignore them here
            pass

        elif opcode_str == 'SWITCH':
            self.make_switch()

        elif opcode_str == 'IF':
            self.make_if(loop)

        elif opcode_str == 'ELIF' or opcode_str == 'ELSE':  # just to avoid them from being added
            pass

        elif opcode_str == 'FOR':
            self.make_for()

        elif opcode_str == 'WHILE':
            self.make_while()

        elif opcode_str == 'DEFINE':
            self.make_define()

        elif opcode_str == 'LCAL':
            self.make_lcal()

        elif opcode_str == 'DW':
            self.make_dw()

        # # # # # # # # # # # # HEADERS # # # # # # # # # # # #

        elif opcode_str == 'IMPORT':
            self.make_import()

        elif opcode_str == 'TEMP':
            inst = Instruction(token(T.word, 'temp'), None)  # Warning: possible for [] to generate undesired code
            self.make_operands(inst)
            for tmp in inst.operands:
                self.add_tmp(tmp)
            self.skip_line()
            return

        # add more later
        # elif opcode_str == '':
        #    self.make_inst()
        else:
            inst = Instruction(opcode, self.get_inst_def(opcode))
            self.make_operands(inst)
            self.add_inst(inst)  # even if the instruction is wrong we still add it to output
            self.check_instruction(inst)

        self.temp = temps  # restore that same information
        self.skip_line()
        return

    def get_inst_def(self, opcode):
        try:
            return self.inst_def[opcode.value.upper()]
        except KeyError:
            self.error(E.unk_instruction, opcode, opcode.value)
            return

    def check_instruction(self, inst: Instruction) -> None:
        if inst.definition is None:  # didnt recognize the instruction, so no point in type checking
            return
        if len(inst.operands) > len(inst.definition.operands):  # we will ignore extra operands, but provide an error
            self.error(E.wrong_op_num, inst.opcode, inst.opcode, len(inst.definition.operands), len(inst.operands))
        if len(inst.operands) == 0:
            return

        types = ot.allowed_types(inst.definition.operands[0], op1=True)
        if types is None or inst.operands[0].type not in types:
            self.error(E.wrong_op_type, inst.operands[0],
                       inst.operands[0].type, inst.definition.operands[0])

        if len(inst.operands) == len(inst.definition.operands) - 1:  # operand shorthands
            inst.operands.insert(1, inst.operands[0])

        # prolly this will change in the future to accommodate for optional operands
        for op, op_def in zip(inst.operands[1:], inst.definition.operands[1:]):
            types = ot.allowed_types(op_def)
            if types is not None and op.type in types:
                continue
            self.error(E.wrong_op_type, op, op.type, op_def)
        return

    def make_operands(self, inst: Instruction, recursive: bool = False) -> List[Token]:
        if recursive:
            while self.has_next() and self.peak().type != T.sym_rpa:
                self.next_operand(inst)
            self.advance()
        else:
            while self.has_next() and self.peak().type != T.newLine:
                self.next_operand(inst)
        return inst.operands

    def next_operand(self, inst: Instruction) -> Token:
        if self.has_next() and self.peak() != T.newLine:
            type = self.peak().type
            value = self.peak().value
            operand = self.peak()

            if type == T.sym_lpa:
                self.advance()
                operand = self.make_instruction(recursive=True)
                if self.has_next() and self.peak().type == T.sym_lbr:
                    if operand is None:
                        self.skip_until(T.sym_rbr)
                        self.advance()

                    elif operand.type not in {T.word, T.string}:  # these are not primitive types
                        self.make_mem_index(inst, operand, operand)
                        inst.operands.append(operand)
                else:
                    inst.operands.append(operand)
                self.set_tmp(operand)
                return operand

            elif type == T.macro:
                if value in default_macros or self.peak(-1).value.upper() == 'DEFINE':  # 1st op of DEFINE
                    pass

                elif value in self.macros:
                    operand = self.macros[value]

                else:
                    self.error(E.undefined_macro, self.peak(), self.peak())
                self.advance()

            elif type == T.label:
                if value not in self.labels:
                    self.error(E.undefined_label, operand, value)
                self.advance()

            elif type == T.string:
                operand = self.make_string(operand)

            elif type == T.array:
                operand = self.make_array(operand, value)

            else:
                self.advance()
            if self.has_next() and self.peak().type == T.sym_lbr:
                temp = self.get_tmp()
                if temp is None:
                    inst.operands.append(operand)
                    self.skip_until(T.sym_rbr)
                    self.advance()

                elif operand.type not in {T.word, T.string}:  # these are not primitive types
                    self.make_mem_index(inst, operand, temp)
                    inst.operands.append(temp)
                    self.set_tmp(temp)
            else:
                inst.operands.append(operand)

            return operand

    def process_scope(self, recursive=False, final_inst=None, start_label=None, end_label=None):
        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            params: List[Instruction, Token, Token] = [final_inst, start_label, end_label]
            self.make_instruction(recursive, params)
        if not self.has_next():
            self.error(E.end_expected, start_label)

    def skip_line(self, inst: Instruction = None, error: bool = False):
        skipped = self.skip_until(T.newLine)
        if error and skipped != 0:
            self.error(E.wrong_op_num, self.peak(-skipped), inst, len(inst.operands), len(inst.operands) + skipped)
        if self.has_next():
            self.advance()
        return

    def skip_until(self, type: T) -> int:
        skipped = 0
        while self.has_next() and self.peak().type is not type:
            skipped += 1
            self.advance()
        return skipped

    def add_tmp(self, tmp: Token) -> None:
        if tmp is not None and tmp.type == T.reg:
            self.temp[tmp] = True

    def get_tmp(self):
        for tmp in self.temp:
            if self.temp[tmp]:
                return tmp
        self.error(E.no_tmp, self.peak())
        return

    def set_tmp(self, tmp: Token):
        self.temp[tmp] = False

    def ret_tmp(self, tmp: Token):
        self.temp[tmp] = True

    def next_word(self):
        while self.has_next() and self.tokens[self.i].type != T.word:
            if self.tokens[self.i].type == T.newLine:
                return  # operands only and not opcode error, ignore this line and proceed
            # Opcode Expected, found operand error
            self.i += 1

        if self.has_next():
            return self.peak()
        else:  # missing Instruction error
            return

    def get_opcode(self):
        while self.has_next() and self.tokens[self.i].type != T.word:
            self.error(E.word_miss, self.tokens[self.i], str(self.tokens[self.i]))
            self.advance()
        if self.has_next():
            return self.peak()
        else:
            self.error(E.word_miss, self.tokens[self.i - 1], 'nothing')
            return

    def make_mem_index(self, inst: Instruction, operand, temp):
        self.advance()
        if self.has_next() and self.peak().type != T.newLine:
            if len(inst.operands) == 0:  # no operands yet -> this is the first operand
                if inst.definition is not None and operand.type in ot.allowed_types(inst.definition.operands[0], True):
                    if inst.definition.operands[0] == 'WB':
                        self.translate_pointer1(inst, operand, temp)

                    elif inst.definition.operands[0] == 'LOC':
                        self.translate_pointer(inst, operand, temp)
            else:
                self.translate_pointer(inst, operand, temp)
            self.advance()
            if self.has_next() and self.peak().type == T.sym_lbr:
                self.make_mem_index(inst, temp, temp)
            self.set_tmp(temp)
            return

    def translate_pointer(self, inst: Instruction, operand, temp):
        if self.peak().type == T.sym_rbr:
            offset = token(T.imm, 0)
        else:
            self.next_operand(inst)
            offset = inst.operands.pop()  # retrieve the operand that got added in the process
        self.skip_until(T.sym_rbr)

        if offset.value != 0:
            self.add_inst(Instruction(token(T.word, 'ADD'), None, temp, operand, offset))
            self.add_inst(Instruction(token(T.word, 'LOD'), None, temp, temp))
        else:
            self.add_inst(Instruction(token(T.word, 'LOD'), None, temp, operand))
        return

    def translate_pointer1(self, inst: Instruction, operand, temp):
        if self.peak().type == T.sym_rbr:
            offset = token(T.imm, 0)
        else:
            self.next_operand(inst)
            offset = inst.operands.pop()  # retrieve the operand that got added in the process
        self.skip_until(T.sym_rbr)
        self.advance()
        if offset.value != '0':
            add_inst = Instruction(token(T.word, 'ADD'), None, operand, operand, offset)
            inst.add_inst_later(add_inst)
            str_inst = Instruction(token(T.word, 'STR'), None, operand, temp)
            add_inst.add_inst_later(str_inst)
            str_inst.add_inst_later(Instruction(token(T.word, 'SUB'), None, operand, operand, offset))
        else:
            inst.add_inst_later(Instruction(token(T.word, 'STR'), None, temp, operand))

        if self.has_next() and self.peak().type == T.sym_lbr:
            self.error(E.operand_expected, self.peak(), self.peak())
            while self.peak().type == T.sym_lbr:
                self.skip_until(T.sym_rbr)
                self.advance()
        self.i -= 1  # rollback 1 advance
        return

    def make_array(self, operand, value) -> Token:
        label = self.label_id + 'array' + str(self.id_count)
        self.id_count += 1
        self.labels.add(label)
        label_tok = Token(T.label, operand.position, operand.line, label)
        self.add_inst(Instruction(label_tok, None))
        self.add_inst(Instruction(token(T.word, 'DW'), None, operand))

        for i, val in enumerate(value):
            if val.type == T.reg:
                value[i] = token(T.imm, 0)
                self.add_inst(Instruction(token(T.word, 'LSTR'), None, label_tok, token(T.imm, i + 1), val))

            elif val.type == T.array:
                value[i] = self.make_array(val, val.value)

            elif val.type == T.string:
                value[i] = self.make_string(val)

        self.advance()
        return label_tok

    def make_string(self, operand) -> Token:
        label = f'{self.label_id}string{str(self.id_count)}'
        self.id_count += 1
        self.labels.add(label)
        label_tok = Token(T.label, operand.position, operand.line, label)
        self.add_inst(Instruction(label_tok, None))
        self.add_inst(Instruction(token(T.word, 'DW'), None, operand))

        self.advance()
        return label_tok

    def make_switch(self) -> None:
        inst = Instruction(Token(T.word, self.peak().position - 1, self.peak().line, 'SWITCH'), None)

        has_default = False
        default_label = None
        end_label, start_label, id = self.get_label_helper(inst)
        case_num = 0
        cases: List[int] = []
        switch: Dict[int] = {}
        dw_loc = len(self.instructions)

        reg: Token = self.next_operand(inst)
        tmp = self.get_tmp()
        self.ret_tmp(tmp)
        self.skip_line(inst, True)
        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            if self.peak().type == T.word and self.peak().value.upper() == 'CASE':
                self.advance()
                operands: List[Token] = self.make_operands(Instruction(token(T.word, 'CASE'), None))
                case_label = token(T.label, f'.reserved_case{id}_{case_num}')
                self.labels.add(case_label.value)
                case_num += 1
                for op in operands:
                    if op.type == T.imm:
                        value = op.value
                    elif op.type == T.char:
                        value = ord(op.value[1:-1])
                    else:
                        self.error(E.invalid_op_type, op, op.type, inst)
                        continue

                    if value in cases:
                        self.error(E.duplicate_case, op)
                    else:
                        cases.append(value)
                        switch[value] = case_label
                self.add_inst(Instruction(case_label, None))

            elif self.peak().type == T.word and self.peak().value.upper() == 'DEFAULT':
                if has_default:
                    self.error(E.duplicate_default, self.peak())
                else:
                    has_default = True
                    default_label = token(T.label, f'.reserved_default{id}')
                    self.labels.add(default_label.value)
                    self.add_inst(Instruction(default_label, None))
                    self.advance()
                    self.skip_line(inst, True)

            elif self.peak().type == T.word and self.peak().value.upper() == 'EXIT':
                self.skip_line(inst, True)
                self.add_inst(Instruction(token(T.word, 'BRA'), None, end_label))

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
                    if has_default:
                        address = default_label
                    else:
                        address = end_label
                dw.append(address)

            smallest_case = token(T.imm, smallest_case)
            biggest_case = token(T.imm, biggest_case)

            dw = str(dw).replace(',', '')
            dw = dw.replace("'", '')
            dw = token(T.word, dw)
            # inserted in reverse order to what they are here
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'DW'), None, dw))
            self.instructions.insert(dw_loc, Instruction(token(T.label, start_label.value), None))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'JMP'), None, tmp))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'LOD'), None, tmp, tmp))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'ADD'), None, tmp, tmp, start_label))
            if smallest_case != 0:
                self.instructions.insert(dw_loc, Instruction(token(T.word, 'SUB'), None, tmp, reg, smallest_case))
            if has_default:
                default_label = end_label
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'SBRG'), None, default_label, reg, biggest_case))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'SBRL'), None, default_label, reg, smallest_case))

        else:
            self.error(E.word_miss, self.peak(), self.peak())

        self.advance()
        return

    def make_if(self, loop=None) -> None:
        inst = Instruction(Token(T.word, self.peak().position - 1, self.peak().line, 'IF'), None)
        else_done = False
        else_count = 0
        end_label, start_label, id = self.get_label_helper(inst)  # end_label, start_label are not used
        next_label: Token = token(T.label, f'.reserved_else{id}_{else_count}')

        self.do_condition(inst, next_label, id)

        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            if self.peak().type == T.word and self.peak().value.upper() == 'ELIF':
                if else_done:
                    self.error(E.missing_if, self.peak(), self.peak())
                else:
                    self.add_inst(Instruction(token(T.word, 'JMP'), None, end_label))
                    self.labels.add(next_label.value)
                    self.add_inst(Instruction(next_label, None))
                    self.advance()
                    else_count += 1
                    next_label = token(T.label, f'.reserved_else{id}_{else_count}')
                    inst.operands = []  # clean the previous if/elif condition operands
                    self.do_condition(inst, next_label, id)

            elif self.peak().type == T.word and self.peak().value.upper() == 'ELSE':
                if else_done:
                    self.error(E.missing_if, self.peak(), self.peak())
                else:
                    else_done = True
                    self.add_inst(Instruction(token(T.word, 'JMP'), None, end_label))
                    self.labels.add(next_label.value)
                    self.add_inst(Instruction(next_label, None))
                    self.advance()
                    self.skip_line(inst, True)
            else:
                self.make_instruction(loop=loop)

        if not else_done:
            self.labels.add(next_label.value)
            self.add_inst(Instruction(next_label, None))
        self.labels.add(end_label.value)
        self.add_inst(Instruction(end_label, None))
        return

    def make_for(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position - 1, self.peak().line, 'FOR'), None)
        end_label, start_label, id = self.get_label_helper(inst)
        reg = self.next_operand(inst)
        end_num = self.next_operand(inst)

        self.labels.add(start_label.value)
        self.add_inst(Instruction(start_label, None))

        if self.has_next() and self.peak().type != T.newLine:  # checking if there is the option parameter
            value_to_add = self.next_operand(inst)
            end_statement: Instruction = Instruction(token(T.word, 'ADD'), None, reg, reg, value_to_add)
        else:
            end_statement: Instruction = Instruction(token(T.word, 'ADD'), None, reg, reg, token(T.imm, 1))

        if end_statement.operands[2].value > 0:
            self.add_inst(Instruction(token(T.word, 'SBGE'), None, end_label, reg, end_num))
        else:
            self.add_inst(Instruction(token(T.word, 'SBLE'), None, end_label, reg, end_num))

        self.skip_line(inst, True)

        self.process_scope(recursive, end_statement, start_label, end_label)

        self.add_inst(end_statement)
        self.add_inst(Instruction(token(T.word, 'JMP'), None, start_label))
        self.labels.add(end_label.value)
        self.add_inst(Instruction(end_label, None))
        return

    def make_while(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position - 1, self.peak().line, 'WHILE'), None)
        end_label, start_label, id = self.get_label_helper(inst)
        self.labels.add(start_label.value)
        self.add_inst(Instruction(start_label, None))

        self.do_condition(inst, end_label, id)

        self.process_scope(recursive, None, start_label, end_label)

        self.add_inst(Instruction(token(T.word, 'JMP'), None, start_label))
        self.labels.add(end_label.value)
        self.add_inst(Instruction(end_label, None))
        return

    def make_lcal(self) -> None:
        inst = Instruction(Token(T.word, self.peak(-1).position, self.peak(-1).line, 'LCAL'), None)
        lib = self.next_operand(inst)
        inst.operands = []  # clean the list, so we can use it for func args
        if lib.type != T.word:
            return
        if lib.value.replace('.', '/') not in self.imported_libs:  # warns it was not imported but translates anyway
            self.error(E.unk_function, self.peak(-1), lib.value)

        if (not self.has_next()) or self.peak().type != T.sym_lpa:
            self.error(E.sym_miss, self.peak(-1), '(')
        else:
            self.advance()
            while self.has_next() and self.peak().type != T.sym_rpa:
                self.next_operand(inst)

            if not self.has_next():
                self.error(E.sym_miss, self.tokens[self.i - 1], ')')

            outs = self.lib_headers[lib.value]['OUTS']
            regs = self.lib_headers[lib.value]['REG']
            for reg_num in range(outs + 1, regs + 1):  # save the registers used not part of the output
                self.add_inst(Instruction(token(T.word, 'PSH'), None, token(T.reg, f'R{reg_num}')))

            for arg in inst.operands[::-1]:  # push the arguments in reverse order
                self.add_inst(Instruction(token(T.word, 'PSH'), None, arg))

            lib.value = '.reserved_' + lib.value.replace('.', '_')
            self.add_inst(Instruction(token(T.word, 'CAL'), None, token(T.label, lib.value)))

            if len(inst.operands) != 0:  # pop args back
                sp = token(T.reg, 'SP')
                self.add_inst(Instruction(token(T.word, 'ADD'), None, sp, sp, token(T.imm, len(inst.operands))))

            for reg_num in range(regs, outs, -1):
                self.add_inst(Instruction(token(T.word, 'POP'), None, token(T.reg, f'R{reg_num}')))

        return

    def process_lib(self, lib_code, lib_name):
        lib_name_replaced = lib_name.replace(".", "_")
        label_id = f'.reserved_{lib_name_replaced}_'
        lexer = Lexer(lib_code, label_id, lib_name.replace('.', '/'))
        lexer.make_tokens()
        parser = Parser(lexer.output, label_id, lib_name.replace('.', '/'), recursive=True)
        parser.inst_def = self.inst_def
        headers = parser.get_lib_headers()
        if self.compare_headers(headers):
            parser.parse()
            self.lib_headers[lib_name] = headers
            dependencies = []
            for lib in parser.imported_libs:
                if lib not in self.imported_libs:  # check and import dependencies
                    dependencies += self.read_lib(lib)

            for error in parser.errors:
                self.errors.append(error)
            label = Instruction(token(T.label, label_id[:-1]), None)
            return [label] + parser.instructions + dependencies, parser.inst_def
        return

    def read_lib(self, name: str):
        path = lib_root + "/" + name.replace(".", "/")
        if os.path.isdir(path):
            lib_code = []
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    if not file.endswith(file_extension):
                        continue
                    lib_file_name = subdir[len(lib_root) + 1:] + '/' + file[:-len(file_extension)]
                    if lib_file_name in self.imported_libs:
                        continue
                    self.imported_libs.add(lib_file_name)
                    with open(os.path.join(subdir, file), 'r') as f:
                        code, inst_defs = self.process_lib(f.read(), lib_file_name.replace('/', '.'))
                        lib_code += code
                        self.inst_def.update(inst_defs)
            return lib_code

        elif os.path.isfile(path + file_extension):
            path += file_extension
            lib_file_name = path[len(lib_root) + 1:-len(file_extension)]
            if lib_file_name in self.imported_libs:
                return
            self.imported_libs.add(lib_file_name)
            with open(path, "r") as f:
                code, inst_defs = self.process_lib(f.read(), lib_file_name.replace('/', '.'))
                self.inst_def.update(inst_defs)
                return code

        elif os.path.isfile(path + file_extensionpp):
            path += file_extensionpp
            lib_file_name = path[len(lib_root) + 1:-len(file_extensionpp)]
            if lib_file_name in self.imported_libs:
                return
            self.imported_libs.add(lib_file_name)
            with open(path, "r") as f:
                code, inst_defs = self.process_lib(f.read(), lib_file_name.replace('/', '.'))
                self.inst_def.update(inst_defs)
                return code

        else:  # this can be used later:   self.error(E.unk_function, self.peak(-1), name.split('.')[1])
            self.error(E.unk_library, self.peak(-1), name)
        return

    def make_import(self):
        inst = Instruction(token(T.word, 'IMPORT'), None)
        self.make_operands(inst)
        for op in inst.operands:
            if op.type == T.word:
                lib_code = self.read_lib(op.value)
                if lib_code is not None:
                    self.lib_code += lib_code
            else:
                self.error(E.wrong_op_type, op, op.type, T.word)
        return

    def compare_headers(self, header) -> bool:
        # TODO
        return True

    def make_define(self) -> None:
        inst = Instruction(Token(T.word, self.peak(-1).position, self.peak().line, 'DEFINE'), None)
        macro = self.next_operand(inst)
        if macro.type != T.macro:
            self.error(E.invalid_op_type, macro, macro.type, 'DEFINE')

        value = self.next_operand(inst)
        # if macro.value in self.macros or macro.value in default_macros:
        #     self.error(E.duplicate_macro, macro, macro.value)
        # else:
        #     self.macros[macro.value] = value
        self.macros[macro.value] = value
        return

    def make_dw(self):
        inst = Instruction(Token(T.word, self.peak().position - 1, self.peak().line, 'DW'), None)
        if self.peak().type == T.string or self.peak().type == T.char:
            string = list(self.peak().value)[1:-1]
            length = len(string)
            string.insert(0, length)
            inst.operands.append(token(T.string, str(string).replace(',', '')))

        elif self.peak().type == T.sym_lbr:
            self.advance()
            self.make_operands(inst)
            args = []
            for element in inst.operands:
                if element.type == T.sym_rbr:
                    break
                if element.type in {T.imm, T.label, T.char, T.mem}:
                    args.append(element.value)
                else:
                    self.error(E.wrong_op_type, element, element.type, 'Imm literal')

            inst.operands = [token(T.word, str(args).replace(',', ''))]

        else:  # error perhaps cause nothing
            self.error(E.sym_miss, self.peak(), '[')
            return

        self.add_inst(inst)
        return

    def get_label_helper(self, inst: Instruction) -> Tuple[Token, Token, int]:
        end_label = token(T.label, f'.reserved_end{self.id_count}')
        loop_label = token(T.label, f'.reserved_{inst.opcode.value}{self.id_count}')
        self.id_count += 1
        return end_label, loop_label, self.id_count - 1

    def do_condition(self, inst: Instruction, label: Token, id):
        def next_expression(operands, count):
            try:
                op = next(operands)
            except StopIteration:
                # should error
                return []

            if op.type in op_precedence:
                val_b = next_expression(operands, count)
                val_a = next_expression(operands, count)

                if op.type == T.sym_and:
                    output = [Instruction(token(T.word, 'PSH'), None, token(T.imm, 0))]
                    next_label = token(T.label, f'{self.label_id}{"next"}_{id}_{next(count)}')

                    if isinstance(val_a, list):
                        if val_a[-1].opcode.value == 'PSH':
                            val_a.pop()
                        else:
                            val_a.append(Instruction(token(T.word, 'POP'), None, val_a))
                        output += val_a
                        val_a = self.get_tmp()

                    elif val_a.type == T.group:
                        mini_parser = Parser(val_a.value, self.label_id, self.file_name, recursive=True)
                        mini_parser.temp = self.temp
                        mini_parser.lib_headers = self.lib_headers
                        val_a = mini_parser.next_operand(Instruction(token(T.word, ''), None))
                        self.instructions += mini_parser.instructions

                    output.append(Instruction(token(T.word, 'BZR'), None, next_label, val_a))

                    if isinstance(val_b, list):
                        if val_b[-1].opcode.value == 'PSH':
                            val_b.pop()
                        else:
                            val.b.append(Instruction(token(T.word, 'POP'), None, val_b))

                        output += val_b
                        val_b = self.get_tmp()

                    elif val_b.type == T.group:
                        mini_parser = Parser(val_b.value, self.label_id, self.file_name, recursive=True)
                        mini_parser.temp = self.temp
                        mini_parser.lib_headers = self.lib_headers
                        val_b = mini_parser.next_operand(Instruction(token(T.word, ''), None))
                        self.instructions += mini_parser.instructions

                    output.append(Instruction(token(T.word, 'STR'), None, token(T.reg, 'SP'), val_b))
                    output.append(Instruction(next_label, None))

                    return output

                elif op.type == T.sym_or:
                    output = [Instruction(token(T.word, 'PSH'), None, token(T.imm, 1))]
                    next_label = token(T.label, f'{self.label_id}{"next"}_{id}_{next(count)}')

                    if isinstance(val_a, list):
                        if val_a[-1].opcode.value == 'PSH':
                            val_a.pop()
                        else:
                            val_a.append(Instruction(token(T.word, 'POP'), None, val_a))
                        output += val_a
                        val_a = self.get_tmp()

                    elif val_a.type == T.group:
                        mini_parser = Parser(val_a.value, self.label_id, self.file_name, recursive=True)
                        mini_parser.temp = self.temp
                        mini_parser.lib_headers = self.lib_headers
                        val_a = mini_parser.next_operand(Instruction(token(T.word, ''), None))
                        self.instructions += mini_parser.instructions

                    output.append(Instruction(token(T.word, 'BNZ'), None, next_label, val_a))

                    if isinstance(val_b, list):
                        if isinstance(val_b, list):
                            if val_b[-1].opcode.value == 'PSH':
                                val_b.pop()
                            else:
                                val.b.append(Instruction(token(T.word, 'POP'), None, val_b))

                            output += val_b
                            val_b = self.get_tmp()

                    elif val_b.type == T.group:
                        mini_parser = Parser(val_b.value, self.label_id, self.file_name, recursive=True)
                        mini_parser.temp = self.temp
                        mini_parser.lib_headers = self.lib_headers
                        val_b = mini_parser.next_operand(Instruction(token(T.word, ''), None))
                        self.instructions += mini_parser.instructions

                    output.append(Instruction(token(T.word, 'STR'), None, token(T.reg, 'SP'), val_b))
                    output.append(Instruction(next_label, None))

                    return output

                else:
                    op_to_inst = {
                        T.sym_equ: 'SETE',
                        T.sym_dif: 'SETNE',
                        T.sym_gt: 'SETGT',
                        T.sym_lt: 'SETLT',
                        T.sym_geq: 'SETGE',
                        T.sym_leq: 'SETLE',
                    }
                    tmp = self.get_tmp()
                    return [Instruction(token(T.word, op_to_inst[op.type]), None, tmp, val_a, val_b),
                            Instruction(token(T.word, 'PSH'), None, tmp)]
            else:
                return op

        operands = self.shunting_yard(inst)
        expression = next_expression(reversed(operands), iter(int, 1))     # infinite iterator

        if isinstance(expression, list):
            cnd = self.get_tmp()
            if expression[-1].opcode.value == 'PSH':
                expression.pop()
            else:
                expression.append(Instruction(token(T.word, 'POP'), None, cnd))
        else:
            cnd = expression

        expression.append(Instruction(token(T.word, 'BRZ'), None, label, cnd))

        self.instructions += expression
        return

    def shunting_yard(self, inst: Instruction) -> List[Token]:
        queue: List[Token] = []
        stack = []
        while self.has_next() and self.peak().type != T.newLine:
            type = self.peak().type
            if type == T.sym_lpa:
                tok = self.peak()
                self.advance()
                if self.peak().type == T.word:
                    queue.append(self.group_inst(tok))
                else:
                    stack.append(tok)

            elif type in op_precedence:
                while len(stack) > 0 and op_precedence[type] > op_precedence[stack[-1].type]:
                    queue.append(stack.pop())
                stack.append(self.peak())
                self.advance()
            elif type == T.sym_rpa:
                while len(stack) > 0 and stack[-1].type != T.sym_lpa:
                    queue.append(stack.pop())
                if len(stack) > 0:
                    stack.pop()
                else:
                    self.error(E.word_miss, self.peak(), 'nothing')
                self.advance()
            else:
                queue.append(self.next_operand(inst))

        if self.has_next():
            while len(stack) > 0:
                if stack[-1].type == T.sym_lpa:
                    self.errors.append(Error(E.sym_miss, self.peak(-1).position, self.peak(-1).line, self.file_name, ')'))
                else:
                    queue.append(stack.pop())
        else:
            self.error(E.word_miss, self.peak(-1), 'nothing')
        self.skip_line()
        return queue

    def group_inst(self, tok) -> Token:
        toks = [tok]
        scope = 0
        while self.has_next() and (self.peak().type != T.sym_rpa or scope != 0):
            tok = self.peak()
            toks.append(tok)
            if tok.type == T.sym_lpa:
                scope += 1
            elif tok.type == T.sym_rpa:
                scope -= 1
            self.advance()
        toks.append(self.peak())
        self.advance()
        return token(T.group, toks)

    def peak(self, j=0) -> Token:
        return self.tokens[self.i + j]

    def next(self) -> Token:
        self.i += 1
        return self.tokens[self.i - 1]

    def advance(self):
        if self.has_next():
            self.i += 1
        else:
            self.error(E.tok_miss, self.peak(-1), 'Nothing')

    def has_next(self, i: int = 0):
        return self.i + i < len(self.tokens)


if __name__ == "__main__":
    main()
