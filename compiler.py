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
    sym_lt = "sum_lt",
    sym_geq = "sym_geq",
    sym_leq = "sum_leq",
    sym_equ = "sym_equ",
    sym_dif = "sum_dif",

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
default_imports = {"inst.core"}


# ERRORS
class E(Enum):
    illegal_char = "Illegal Char '{}'"
    invalid_char = "Invalid Character {}"
    unk_port = "Unknown port name '{}'"
    miss_pair = "Missing closing quote {}"
    word_miss = "Keyword expected, found {} instead"
    tok_miss = "Token expected, found {} instead"

    def __repr__(self) -> str:
        return self.value


usage = """usage: urclpp <source_file> <destination_file>"""


def main():
    source_name = argv[1] if len(argv) >= 2 else None  
    dest_name = argv[2] if len(argv) >= 3 else None

    source = r'''INST ADD[4][]'''
    
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

    tokens, lex_errors = Lexer(source).make_tokens()

    print("tokens:", file=dest)
    print(tokens, file=dest)
    print("\n", file=dest)
    
    if len(lex_errors) > 1:
        print(lex_errors, file=stderr)
        exit(1)
    # parse
    instructions, parse_errors = Parser(tokens).parse()

    print("program:", file=dest)
    print(instructions, file=dest)
    print("\n", file=dest)

    if len(parse_errors) > 1:
        print(parse_errors, file=stderr)
        exit(1)

    return


class Token:
    def __init__(self, type: T, pos: int, line: int, value: str = "") -> None:
        self.type = type
        self.position = pos
        self.line = line
        self.macro_type = None  # in parsing stage this will be assigned the type the macro is replacing
        self.value = value
        pass
    
    def __repr__(self) -> str:
        return f"<{self.type} {self.value}>"


class Error:
    def __init__(self, error: E, index: int, line: int, extra: str = "") -> None:
        self.error = error
        self.line = line
        self.index = index
        self.extra = extra

    def __repr__(self) -> str:
        return f'{self.error.value.format(self.extra)}, at {self.index} at line {self.line}'


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
                    self.new_line()
                    self.advance()

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
            return invalid_char
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
        return self.i + i < len(self.p)


@dataclass
class Id:
    pass

class OT(Enum):
    Read = "read"
    Write = "write"
    Imm = "imm"
    Any = "any"

op_types = {
    'R': OT.Read,
    'W': OT.Write,
    'I': OT.Imm,
    'A': OT.Any,
}

@dataclass
class InstDef(Id):
    operands: OrderedDict[str, OT]

@dataclass
class LabelDef(Id):
    location: int


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
        self.tokens = tokens
        self.ids: dict[str, Id] = {}
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []
        self.i = 0
    

    def error(self, error: E, token: Token, extra: str = ""):
        self.errors.append(Error(error, token.position, token.line, extra))
    def error_str(self, msg: str):
        # TODO make this make sense lol
        token = self.peak() if self.has_next() else self.tokens[-1]
        self.errors.append(Error(E.illegal_char, token.position, token.line, msg))

    def parse(self):
        while self.has_next():
            word = self.next_word()
            if (word is None):
                break

            if word.value.upper() == 'INST':
                self.make_inst_def()

            self.make_instruction()
            # To prevent infinite loop, we should probably check whether a token was found and report an error instead 
            self.i += 1
        print(self.ids)
        return self.instructions, self.errors

    def make_inst_def(self) -> None:
        self.i += 1
        name = self.next_word()
        if (name is None):
            self.error(E.tok_miss, self.tokens[-1])
            return

        inst = InstDef(OrderedDict())
        if self.ids.get(name.value) is not None:
            self.error_str(f"identifier {name} is already defined")
            return
        self.ids[name.value] = inst


        self.i += 1
        while (self.has_next() and self.tokens[self.i].type is not T.newLine):
            op_name = self.next()
            if op_name is None:
                self.error_str(msg="missing operant name")
                self.skip_line()
                return
            if not self.has_next() or self.next().type is not T.sym_col:
                self.error_str(msg="missing colon")
                self.skip_line()
                return

            type_str = self.next()
            op_type = None if type_str is None else op_types.get(type_str.value)
            if op_type is None:
                self.error_str(msg="missing type (R, W, I or A)")
                return
            inst.operands[op_name.value] = op_type
        
        pass

    def make_instruction(self, recursive: bool = False) -> None:
        opcode = self.next_word()
        if opcode is None:
            return
        # needs to check if number of operands are correct via dict/enum with expected ones

        self.instructions.append(Instruction(opcode))
        return

    def process_scope(self):  # calls make instructions for the rest of the scope below

        return
    
    def peak(self):
        return self.tokens[self.i]
    
    def next(self):
        self.i += 1
        return self.tokens[self.i-1]

    def skip_line(self):
        if self.skip_until(T.newLine): self.i += 1

    def skip_until(self, type: T):
        while self.has_next() and self.next().type is not type: pass
        return self.has_next()

    def get_opcode(self):
        while self.has_next() and self.tokens[self.i].type != T.word:
            self.error(E.word_miss, self.tokens[self.i], str(self.tokens[self.i]))
            self.advance()

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
