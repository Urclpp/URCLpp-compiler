from os.path import isfile
from sys import argv, stdout, stderr
from enum import Enum
from typing import List, Tuple

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
}
opcodes = {

}
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}
default_imports = {"inst.core"}

# ERRORS
illegal_char = "Illegal Char '{}' at line {}\n"
invalid_char = "Invalid Character {} at line {}\n"
unk_port = "Unknown port name '{}' at line {}\n"
miss_pair = "Missing closing quote {} at line {}\n"

usage = """usage: urclpp <source_file> <destination_file>"""


def main():
    source_name = argv[1] if len(argv) >= 2 else None  
    dest_name = argv[2] if len(argv) >= 3 else None

    source = r'''.yeet @ohboi[4][2] ahah'''
    
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
    
    if lex_errors != '':
        print(lex_errors, file=stderr)
        exit(1)
    # parse
    instructions, parse_errors = Parser(tokens).parse()

    print("program:", file=dest)
    print(instructions, file=dest)
    print("\n", file=dest)

    if parse_errors != '':
        print(parse_errors, file=stderr)
        exit(1)

    return


class Token:
    def __init__(self, type: T, value: str = "") -> None:
        self.type = type
        self.macro_type = None  # in parsing stage this will be assigned the type the macro is replacing
        self.value = value
        pass
    
    def __repr__(self) -> str:
        return f"<{self.type} {self.value}>"


class Lexer:
    def __init__(self, program: str) -> None:
        self.p = program
        self.line_nr = 0
        self.i = 0
        self.output: list[Token] = []
        self.errors = ''

    def token(self, type: T, value: str = "") -> None:
        self.output.append(Token(type, value))

    def make_tokens(self):
        while self.has_next():
            while self.has_next() and self.p[self.i] in ' ,\t':  # ignore commas and indentation
                self.i += 1
            if self.p[self.i] == '\n':  # change line
                self.i += 1
                self.line_nr += 1

            elif self.p[self.i] == '/':
                self.i += 1
                if self.has_next() and self.p[self.i] == '/':  # inline comment
                    self.inline_comment()
                elif self.has_next() and self.p[self.i] == '*':
                    self.multi_line_comment()
                else:  # you got your hopes high but it was just an illegal char :/
                    self.errors += illegal_char.format('/', self.line_nr)

            elif self.p[self.i] in symbols:
                self.token(symbols[self.p[self.i]])
                self.i += 1
            else:
                self.make_operand()

        return self.output, self.errors

    def make_operand(self, indexed: bool = False) -> None:
        if self.p[self.i] in digits + '+-':  # immediate value
            self.token(T.imm, str(self.make_num(indexed)))

        elif self.p[self.i] in charset:  # opcode
            self.token(T.word, self.make_word(indexed))

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.i += 1
            if prefix in 'rR$':  # register
                if self.p[self.i] not in '+-':
                    Token(T.reg, 'R' + str(self.make_num(indexed)))
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix in 'mM#':  # memory
                if self.p[self.i] not in '+-':
                    Token(T.mem, 'M' + str(self.make_num(indexed)))
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.token(T.port, '%' + str(self.make_num()))
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.token(T.port, '%' + name)
                    else:
                        self.errors += unk_port.format(name, self.line_nr)

            elif prefix == '~':  # relative
                self.token(T.relative, prefix + str(self.make_num(indexed)))

            elif prefix == '.':  # label
                self.token(T.label, prefix + self.make_word(indexed))

            elif prefix == '@':  # macro
                self.token(T.macro, prefix + self.make_word(indexed))

            elif prefix == "'":  # character
                char = self.make_str("'")
                if char == invalid_char:
                    pass
                elif len(char) == 3:  # char = "'<char>'"
                    self.token(T.char, char)
                else:
                    self.errors += invalid_char.format(char, self.line_nr)

            elif prefix == '"':
                self.token(T.string, self.make_str('"'))

            # elif prefix == '':
            #    self.token()

            else:  # unknown symbol
                if indexed and self.p[self.i] == ']':
                    self.i += 1
                else:
                    self.errors += illegal_char.format(self.p[self.i-1], self.line_nr)

    def make_str(self, char: str) -> str:
        word = char
        while self.has_next() and self.p[self.i] != char and self.p[self.i] != '\n':
            word += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.token(T.newLine)
            self.errors += miss_pair.format(char, self.line_nr)
            return invalid_char
        else:
            word += self.p[self.i]
            self.i += 1
            return word

    def make_word(self, indexed: bool = False) -> str:
        word = self.p[self.i]
        self.i += 1
        while self.has_next() and self.p[self.i] not in indentation:
            if self.p[self.i] == '[':  # has pointer after the operand
                self.make_mem_index()
                return word
            elif indexed and self.p[self.i] == ']':
                self.i += 1
                return word

            if self.p[self.i] not in charset:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                word += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.token(T.newLine)
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
                self.i += 1
                return num

            if self.p[self.i] not in digits + bases:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                num += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.token(T.newLine)
        return int(num, 0)

    def make_mem_index(self) -> None:
        self.i += 1
        while self.p[self.i] == ' ':
            self.i += 1
        if self.p[self.i] == ']':
            self.token(T.pointer, Token(T.imm, '0'))
            self.i += 1
        else:
            self.make_operand(True)     # make the operand at the index and push the token to output
            index = self.output.pop()   # retrieve that token generated
            self.token(T.pointer, index)

        if self.p[self.i] == '[':
            self.make_mem_index()

    def multi_line_comment(self) -> None:
        while self.has_next(1) and (self.p[self.i] != '*' or self.p[self.i + 1] != '/'):
            if self.p[self.i] == '\n':
                self.line_nr += 1
            self.i += 1
        self.i += 2

    def inline_comment(self) -> None:
        while self.has_next() and self.p[self.i] != '\n':
            self.i += 1
        self.i += 1
        self.line_nr += 1

    def has_next(self, i: int = 0) -> bool:
        return self.i + i < len(self.p)


@dataclass
class Id:
    pass


class OT(Enum):
    Reg = "reg"
    Imm = "imm"
    Any = "any"
    pass


class OpOp(Enum):
    read = "read"
    write = "write"
    both = "both"


@dataclass
class OperantDef:
    type: OT
    op: OpOp


@dataclass
class InstDef(Id):
    operands: List[OperantDef]


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
        self.inst_defs: dict[str, InstDef] = {}
        self.instructions: list[Instruction] = []
        self.errors = ''
        self.i = 0

    def parse(self) -> Tuple[List[Instruction], str]:
        while self.has_next():
            self.make_instruction()
            # To prevent infinite loop, we should probably check whether a token was found and report an error instead 
            self.i += 1

        return self.instructions, self.errors

    def make_instruction(self) -> None:
        opcode = self.next_word()
        if opcode is None:
            return
        # needs to check if number of operands are correct via dict/enum with expected ones

        self.instructions.append(Instruction(opcode))
        return

    def process_scope(self):  # calls make instructions for the rest of the scope below

        return

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

    def has_next(self, i: int = 0):
        return self.i + i < len(self.tokens)


if __name__ == "__main__":
    main()
