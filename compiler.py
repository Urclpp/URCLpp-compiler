import os
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
    char = 'opr_cha'
    string = 'opr_str'
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
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}
default_macros = {'@BITS', '@MINREG', '@MINRAM', '@HEAP', '@MINSTACK', '@MSB', '@SMSB', '@MAX', '@SMAX', '@UHALF',
                  '@LHALF'}

file_extension = '.urcl'
lib_root = 'urclpp-libraries'
default_imports = set()  # default_imports = {"inst.core", "inst.io", "inst.basic", "inst.complex"}


# ERRORS
class E(Enum):
    illegal_char = "Illegal Char '{}'"
    invalid_char = "Invalid Character {}"
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
    undefined_macro = "Undefined macro '{}' used"
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

    source = r'''
'''

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
    for inst in parser.instructions:
        print(inst, file=dest)
    # print(parser.instructions, file=dest)
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
        self.type: T = type
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
        return f'{self.error.value.format(*self.args)}, at char {self.index} at line {self.line}'


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
                    else:  # you got your hopes high, but it was just an illegal char :/
                        self.error(E.illegal_char, '/')

                elif self.p[self.i] in symbols or self.p[self.i] in {'=', '!'}:    # gotta check for first char of all
                    self.make_symbol()
                    self.advance()
                else:
                    self.make_operand()

        return self.output, self.errors

    def make_symbol(self):
        if self.p[self.i] == '<':
            if self.has_next(1) and self.p[self.i+1] == '=':
                self.token(symbols['<='])
                self.advance()
            else:
                self.token(symbols['<'])

        elif self.p[self.i] == '>':
            if self.has_next(1) and self.p[self.i+1] == '=':
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

        elif self.p[self.i] == '&' and self.has_next(1) and self.p[self.i+1] == '&':
            self.advance()
            self.token(T.sym_and)

        elif self.p[self.i] == '|' and self.has_next(1) and self.p[self.i+1] == '|':
            self.advance()
            self.token(T.sym_or)

        else:
            try:
                self.token(symbols[self.p[self.i]])
            except KeyError:
                self.error(E.illegal_char, self.p[self.i])

    def make_operand(self, indexed: bool = False) -> None:
        if self.p[self.i] in digits + '+-':  # immediate value
            self.token(T.imm, str(self.make_num()))

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.advance()

            if prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.token(T.port, '%' + str(self.make_num()))
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.token(T.port, '%' + name)
                    else:
                        self.error(E.unk_port, name)

            elif prefix == '~':  # relative
                self.token(T.relative, prefix + str(self.make_num()))

            elif prefix == '.':  # label
                self.token(T.label, prefix + self.make_word())

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
                if string == 'SP' or string == 'PC':
                    self.token(T.reg, string)
                elif prefix in 'mM#rR$':
                    try:
                        if prefix in 'rR$':  # register
                            self.token(T.reg, 'R' + str(int(string, 0)))
                        else:  # memory
                            self.token(T.mem, 'M' + str(int(string, 0)))
                    except ValueError:
                        self.token(T.word, string)
                else:
                    self.token(T.word, string)

            # elif prefix == '':
            #    self.token()

            else:  # unknown symbol
                self.error(E.illegal_char, self.p[self.i-1])

        if self.has_next() and self.p[self.i] == '\n':
            self.new_line()
            self.advance()

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

    def make_num(self) -> int:
        if self.p[self.i] == ' ':
            return 0
        num = ''
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] in symbols:
                break
            elif self.p[self.i] not in digits + bases:
                self.error(E.illegal_char, self.p[self.i])
            else:
                num += self.p[self.i]
            self.advance()
        return int(num, 0)

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


class OpType(Enum):
    R = 'Reg'
    I = 'Imm'
    W = 'Write'
    M = 'Mem'
    A = 'Any'
    P = 'IO'

    def __repr__(self) -> str:
        return self.value


class InstDef:
    def __init__(self, opcode: Token, *args: Token) -> None:
        self.opcode = opcode
        self.operands: list[OpType] = []
        for op in args:
            self.operands.append(OpType[op])    # pre: check dont pass a None or an invalid key. need to check b4

    def __repr__(self) -> str:
        out = f'<INSTDEF {self.opcode}'
        for op in self.operands:
            out += ' ' + str(op)
        return out + '>'


operand1_type = {  # used for the first operand
    T.label: OpType.M,
    T.reg: OpType.W,
    T.imm: OpType.M,
    T.char: OpType.I,
    T.mem: OpType.M,
    T.port: OpType.P,
    T.relative: OpType.M,
}
operand_type = {
    T.label: OpType.I,
    T.reg: OpType.R,
    T.imm: OpType.I,
    T.char: OpType.I,
    T.mem: OpType.I,
    T.port: OpType.I,
    T.relative: OpType.I,
}


class Instruction:
    def __init__(self, opcode: Token, inst_def, *args: Token) -> None:
        self.opcode = opcode
        self.operands: list[Token] = []
        self.definition: InstDef = inst_def
        for op in args:
            if op is not None:
                self.operands.append(op)

    def __repr__(self) -> str:
        out = f'<INST {self.opcode}'
        for op in self.operands:
            out += ' ' + str(op)
        return out + '>'


def token(type: T, value=''):
    return Token(type, -1, -1, value)


class Parser:
    def __init__(self, tokens: List[Token], recursive: bool = False):
        self.tokens: list[Token] = tokens
        self.ids: dict[str, Id] = {}
        self.instructions: list[Instruction] = []
        self.inst_def: dict[InstDef] = {
            'ADD': InstDef('ADD', 'W', 'R', 'R')
        }
        self.errors: list[Error] = []

        self.temp: dict[Token] = {token(T.reg, 'tmp'): True, token(T.reg, 'tmp2'): True}
        self.id_count = 0
        self.macros: dict[Token] = {}
        self.labels: set(str) = set()
        self.lib_headers: dict[str] = {}
        self.imported_libs = set()
        self.recursive = recursive
        if recursive:
            self.lib_code = []
        else:
            self.lib_code = [Instruction(token(T.word, 'HLT'), None)]
        self.i = 0

    def error(self, error: E, tok: Token, *args):
        self.errors.append(Error(error, tok.position, tok.line, *args))

    def add_inst(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def make_inst_def(self):
        # TODO
        return

    def get_lib_headers(self):
        headers = {
            'BITS': None,
            'OUTS': None,
            'OPS': None,
            'REG': None
        }
        while self.has_next():
            header = self.get_opcode()
            if header.value in headers:
                self.advance()
                inst = Instruction(header, None)
                if inst.opcode.value in {'BITS'}:
                    comparison_op = self.next_operand(inst)
                op = self.next_operand(inst)
                if op is not None and op.type == T.imm:
                    headers[header.value] = int(op.value, 0)
            else:   # headers are over
                if self.i != 0:
                    self.i -= 1
                else:
                    pass    # should get error cause no headers were provided
                break
        return headers

    def parse(self):
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
        self.advance()
        if not self.has_next():
            return

        temps: Dict[Token] = self.temp.copy()   # save the current state of the temps

        opcode_str = opcode.value.upper()
        if recursive:
            inst = Instruction(opcode, self.get_inst_def(opcode))
            tmp = self.get_tmp()
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
                if loop[0] is not None:     # while loops dont have a final statement
                    self.add_inst(loop[0])
                self.add_inst(Instruction(token(T.word, 'JMP'), None, loop[1]))

        elif opcode_str == 'END':   # ends are recognized by the funcs that use them, so we will ignore them here
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

        # # # # # # # # # # # # HEADERS # # # # # # # # # # # #

        elif opcode_str == 'IMPORT':
            self.make_import()

        elif opcode_str == 'TEMP':
            inst = Instruction(token(T.word, 'temp'), None)   # Warning: possible for [] to generate undesired code
            self.make_operands(inst)
            for tmp in inst.operands:
                self.add_tmp(tmp)

        # add more later
        # elif opcode_str == '':
        #    self.make_inst()
        else:
            inst = Instruction(opcode, self.get_inst_def(opcode))
            self.make_operands(inst)
            self.instructions.append(inst)  # even if the instruction is wrong we still add it to output

            self.check_instruction(inst)

        self.temp = temps   # restore that same information
        self.skip_line()
        return

    def get_inst_def(self, opcode):
        try:
            return self.inst_def[opcode.value]
        except KeyError:
            self.error(E.unk_instruction, opcode, opcode)
            return

    def check_instruction(self, inst: Instruction) -> None:
        if inst.definition is None:     # didnt recognize the instruction, so no point in type checking
            return
        if len(inst.operands) > len(inst.definition.operands):     # we will ignore extra operands, but provide an error
            self.error(E.wrong_op_num, inst.opcode, inst.opcode, len(inst.definition.operands), len(inst.operands))
        if len(inst.operands) == 0:
            return

        if operand1_type[inst.operands[0].type] != inst.definition.operands[0]:
            self.error(E.wrong_op_type, inst.operands[0], inst.operands[0], inst.definition.operands[0])

        if len(inst.operands) == len(inst.definition.operands)-1:  # operand shorthands
            inst.operands.insert(1, inst.operands[0])

        # prolly this will change in the future to accommodate for optional operands
        for op, op_def in zip(inst.operands[1:], inst.definition.operands[1:]):
            if op_def != 'A' and operand_type[op.type] != op_def:
                self.error(E.wrong_op_type, op, op, op_def)

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
            operand = None

            if type == T.sym_lpa:
                self.advance()
                operand = self.make_instruction(recursive=True)
                if self.has_next() and self.peak().type == T.sym_lbr:
                    if operand is not None and operand.type not in {T.word, T.string}:  # these are not primitive types
                        self.make_mem_index(inst, operand, operand)
                        inst.operands.append(operand)
                else:
                    inst.operands.append(operand)
                self.set_tmp(operand)
                return operand

            elif type == T.macro:
                if value in self.macros:
                    operand = self.macros[value]

                elif value in default_macros:
                    operand = self.peak()

                elif inst.opcode.value.upper() == 'DEFINE' and 'DEFINE' == self.peak(-1).value:  # 1st op of DEFINE
                    operand = self.peak()

                else:
                    self.error(E.undefined_macro, self.peak(), self.peak())
                self.advance()

            else:
                operand = self.peak()
                self.advance()
            if self.has_next() and self.peak().type == T.sym_lbr:
                temp = self.get_tmp()
                if temp is not None and operand.type not in {T.word, T.string}:  # these are not primitive types
                    self.make_mem_index(inst, operand, temp)
                    self.ret_tmp(temp)
                    inst.operands.append(temp)
            else:
                inst.operands.append(operand)

            return operand

    def process_scope(self, recursive=False, final_inst=None, start_label=None, end_label=None):
        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            params: list[Instruction, Token, Token] = [final_inst, start_label, end_label]
            self.make_instruction(recursive, params)
        if not self.has_next():
            self.error(E.end_expected, start_label)

    def skip_line(self, inst: Instruction = None, error: bool = False) -> int:
        skipped = self.skip_until(T.newLine)
        if error and skipped != 0:
            self.error(E.wrong_op_num, self.peak(-skipped), inst, len(inst.operands), len(inst.operands)+skipped)
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
            self.temp[Token] = True

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
            self.error(E.word_miss, self.tokens[self.i-1], 'nothing')
            return

    def make_mem_index(self, inst: Instruction, operand, temp):
        self.advance()
        if self.has_next() and self.peak().type != T.newLine:
            if len(inst.operands) == 0:  # no operands yet -> this is the first operand
                if operand.type in operand1_type:
                    if inst.definition.operands[0] == OpType.W:
                        # self.add_inst(Instruction(token(T.word, 'STR'), None, temp, <dest reg here>))
                        pass
                    elif inst.definition.operands[0] == OpType.M:
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
            offset = token(T.imm, '0')
        else:
            self.next_operand(inst)
            offset = inst.operands.pop()  # retrieve the operand that got added in the process
        self.skip_until(T.sym_rbr)

        if offset.value != '0':
            self.add_inst(Instruction(token(T.word, 'ADD'), None, temp, operand, offset))
            self.add_inst(Instruction(token(T.word, 'LOD'), None, temp, temp))
        else:
            self.add_inst(Instruction(token(T.word, 'LOD'), None, temp, operand))
        return

    def make_switch(self) -> None:
        inst = Instruction(Token(T.word, self.peak().position-1, self.peak().line, 'SWITCH'), None)

        has_default = False
        default_label: str = None
        end_label, start_label, id = self.get_label_helper(inst)
        case_num = 0
        cases: List[int] = []
        switch: dict[int] = {}
        dw_loc = len(self.instructions)

        reg: Token = self.next_operand(inst)
        tmp = self.get_tmp()
        self.ret_tmp(tmp)
        self.skip_line(inst, True)
        while self.has_next() and (self.peak().type != T.word or self.peak().value.upper() != 'END'):
            if self.peak().type == T.word and self.peak().value.upper() == 'CASE':
                self.advance()
                operands: List[Token] = self.make_operands(Instruction(token(T.word, 'CASE'), None))
                case_label = f'.reserved_case{id}_{case_num}'
                self.labels.add(case_label)
                case_num += 1
                for op in operands:
                    if op.type == T.imm:
                        value = int(op.value, 0)
                        if value in cases:
                            self.error(E.duplicate_case, op)
                        else:
                            cases.append(value)
                            switch[value] = case_label
                            self.add_inst(Instruction(token(T.label, case_label), None))
                    else:
                        self.error(E.invalid_op_type, op, op.type, inst)

            elif self.peak().type == T.word and self.peak().value.upper() == 'DEFAULT':
                if has_default:
                    self.error(E.duplicate_default, self.peak())
                else:
                    has_default = True
                    default_label = f'.reserved_default{id}'
                    self.labels.add(default_label)
                    self.add_inst(Instruction(token(T.label, default_label), None))
                    self.advance()
                    self.skip_line(inst, True)

            elif self.peak().type == T.word and self.peak().value.upper() == 'EXIT':
                self.skip_line(inst, True)
                self.add_inst(Instruction(Token(T.word, pos, line, 'BRA'), None, end_label))

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

            dw = str(dw).replace(',', '')
            dw = dw.replace("'", '')
            # inserted in reverse order to what they are here
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'DW'), None, dw))
            self.instructions.insert(dw_loc, Instruction(token(T.label, start_label), None))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'JMP'), None, tmp))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'LOD'), None, tmp, tmp))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'ADD'), None, tmp, tmp, start_label))
            if smallest_case != 0:
                self.instructions.insert(dw_loc, Instruction(token(T.word, 'SUB'), None, tmp, reg, smallest_case))
            if has_default:
                default_label = end_label
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'BRG'), None, default_label, reg, biggest_case))
            self.instructions.insert(dw_loc, Instruction(token(T.word, 'BRL'), None, default_label, reg, smallest_case))

        else:
            self.error(E.word_miss, self.peak(), self.peak())

        self.advance()
        return

    def make_if(self, loop=None) -> None:
        inst = Instruction(Token(T.word, self.peak().position-1, self.peak().line, 'IF'), None)
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
                    self.labels.add(next_label)
                    self.add_inst(Instruction(next_label, None))
                    self.advance()
                    else_count += 1
                    next_label = token(T.label, f'.reserved_else{id}_{else_count}')
                    inst.operands = []      # clean the previous if/elif condition operands
                    self.do_condition(inst, next_label, id)

            elif self.peak().type == T.word and self.peak().value.upper() == 'ELSE':
                if else_done:
                    self.error(E.missing_if, self.peak(), self.peak())
                else:
                    else_done = True
                    self.add_inst(Instruction(token(T.word, 'JMP'), None, end_label))
                    self.labels.add(next_label)
                    self.add_inst(Instruction(next_label, None))
                    self.advance()
                    self.skip_line(inst, True)
            else:
                self.make_instruction(loop=loop)

        if not else_done:
            self.labels.add(next_label)
            self.add_inst(Instruction(next_label, None))
        self.labels.add(end_label)
        self.add_inst(Instruction(end_label, None))

        # self.advance()
        return

    def make_for(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position-1, self.peak().line, 'FOR'), None)
        end_label, start_label, id = self.get_label_helper(inst)
        reg = self.next_operand(inst)
        end_num = self.next_operand(inst)

        self.labels.add(start_label)
        self.add_inst(Instruction(start_label, None))
        self.add_inst(Instruction(token(T.word, 'BGE'), end_label, reg, end_num))

        if self.has_next() and self.tokens[self.i] != T.newLine:  # checking if there is the option parameter
            value_to_add = self.next_operand(inst)
            end_statement: Instruction = Instruction(token(T.word, 'ADD'), None, reg, reg, value_to_add)
        else:
            end_statement: Instruction = Instruction(token(T.word, 'ADD'), None, reg, reg, token(T.imm, '1'))
        self.skip_line(inst, True)

        self.process_scope(recursive, end_statement, start_label, end_label)

        self.add_inst(end_statement)
        self.add_inst(Instruction(token(T.word, 'JMP'), None, start_label))
        self.labels.add(end_label)
        self.add_inst(Instruction(end_label, None))
        return

    def make_while(self, recursive=False) -> None:
        inst = Instruction(Token(T.word, self.peak().position-1, self.peak().line, 'WHILE'), None)
        end_label, start_label, id = self.get_label_helper(inst)
        self.labels.add(start_label)
        self.add_inst(Instruction(start_label, None))

        self.do_condition(inst, end_label, id)

        self.process_scope(recursive, None, start_label, end_label)

        self.add_inst(Instruction(token(T.word, 'JMP'), None, start_label))
        self.labels.add(end_label)
        self.add_inst(Instruction(end_label, None))
        return

    def make_lcal(self) -> None:
        inst = Instruction(Token(T.word, self.peak(-1).position, self.peak(-1).line, 'LCAL'), None)
        lib = self.next_operand(inst)
        inst.operands = []      # clean the list, so we can use it for func args
        if lib.type != T.word:
            return
        if lib.value.replace('.', '/') not in self.imported_libs:   # warns it was not imported but translates anyway
            self.error(E.unk_function, self.peak(-1), lib.value)

        if (not self.has_next()) or self.peak().type != T.sym_lpa:
            self.error(E.sym_miss, self.peak(-1), '(')
        else:
            self.advance()
            while self.has_next() and self.peak().type != T.sym_rpa:
                self.next_operand(inst)

            if self.has_next():
                self.skip_line()
            else:
                self.error(E.sym_miss, self.tokens[self.i - 1], ')')

            outs = self.lib_headers[lib.value]['OUTS']
            regs = self.lib_headers[lib.value]['REG']
            for reg_num in range(outs + 1, regs + 1):      # save the registers used not part of the output
                self.add_inst(Instruction(token(T.word, 'PSH'), None, token(T.reg, f'R{reg_num}')))

            for arg in inst.operands[::-1]:     # push the arguments in reverse order
                self.add_inst(Instruction(token(T.word, 'PSH'), None, arg))

            lib.value = '.reserved_' + lib.value
            self.add_inst(Instruction(token(T.word, 'CAL'), None, token(T.label, lib.value)))

            if len(inst.operands) != 0:     # pop args back
                sp = token(T.reg, 'SP')
                self.add_inst(Instruction(token(T.word, 'ADD'), None, sp, sp, token(T.imm, len(inst.operands))))

            for reg_num in range(regs, outs, -1):
                self.add_inst(Instruction(token(T.word, 'POP'), None, token(T.reg, f'R{reg_num}')))

        return

    def process_lib(self, lib_code, lib_name):
        lexer = Lexer(lib_code)
        lexer.make_tokens()
        parser = Parser(lexer.output, True)
        headers = parser.get_lib_headers()
        if self.compare_headers(headers):
            parser.parse()
            self.lib_headers[lib_name] = headers
            return [Instruction(token(T.label, '.reserved_' + lib_name), None)] + parser.instructions
        return

    def read_lib(self, name: str):
        path = lib_root + "/" + name.replace(".", "/")
        if os.path.isdir(path):
            lib_code = []
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    lib_file_name = subdir[len(lib_root)+1:] + '/' + file[:-len(file_extension)]
                    if lib_file_name in self.imported_libs:
                        continue
                    self.imported_libs.add(lib_file_name)
                    with open(os.path.join(subdir, file), 'r') as f:
                        lib_code += self.process_lib(f.read(), lib_file_name.replace('/', '.'))
            return lib_code
        path += file_extension
        if os.path.isfile(path):
            lib_file_name = path[len(lib_root) + 1:-len(file_extension)]
            if lib_file_name in self.imported_libs:
                return
            self.imported_libs.add(lib_file_name)
            with open(path, "r") as f:
                return self.process_lib(f.read(), lib_file_name.replace('/', '.'))

        else:   # this can be used later:   self.error(E.unk_function, self.peak(-1), name.split('.')[1])
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
        if macro.value in self.macros or macro.value in default_macros:
            self.error(E.duplicate_macro, macro, macro)
        else:
            self.macros[macro.value] = value
        return

    def get_label_helper(self, inst: Instruction) -> Token:
        end_label = token(T.label, f'.reserved_end{self.id_count}')
        loop_label = token(T.label, f'.reserved_{inst.opcode.value}{self.id_count}')
        self.id_count += 1
        return end_label, loop_label, self.id_count-1

    def do_condition(self, inst: Instruction, label: Token, id):
        operands = self.make_operands(inst)
        self.skip_line()
        if len(operands) == 1:
            self.add_inst(Instruction(token(T.word, 'BRZ'), None, label, operands[0]))
        elif len(operands) == 3:
            cnd = operands[1].type
            comparison = {
                T.sym_lt: Instruction(token(T.word, 'BGE'), None, label, operands[0], operands[2]),
                T.sym_gt: Instruction(token(T.word, 'BLE'), None, label, operands[0], operands[2]),
                T.sym_geq: Instruction(token(T.word, 'BRL'), None, label, operands[0], operands[2]),
                T.sym_leq: Instruction(token(T.word, 'BRG'), None, label, operands[0], operands[2]),
                T.sym_dif: Instruction(token(T.word, 'BRE'), None, label, operands[0], operands[2]),
                T.sym_equ: Instruction(token(T.word, 'BNE'), None, label, operands[0], operands[2]),
            }
            if cnd not in comparison:
                return  # wrong condition or smt
            else:
                self.add_inst(comparison[cnd])
        # self.add_inst(Instruction(token(T.label, f'.reserved_body{id}')))
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

    def peak(self, j=0) -> Token:
        return self.tokens[self.i+j]

    def next(self) -> Token:
        self.i += 1
        return self.tokens[self.i-1]

    def advance(self):
        if self.has_next():
            self.i += 1
        else:
            self.error(E.tok_miss, self.peak(-1), 'Nothing')

    def has_next(self, i: int = 0):
        return self.i + i < len(self.tokens)


if __name__ == "__main__":
    main()
