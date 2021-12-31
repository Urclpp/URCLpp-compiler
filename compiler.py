# CONSTANTS
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
digits = '1234567890'
bases = 'oOxXbB'
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}

# ERRORS
illegal_char = "Illegal Char '{}' at line {}\n"
unk_operand = "Unknown operand type at line {}\n"
unk_port = "Unknown port name/number at line {}\n"


def main():
    test = Lexer('-0X45')
    print(test.make_operand())
    print(test.errors)
    return


class Lexer:
    def __init__(self, program):
        self.p = program
        self.line_nr = 0
        self.i = 0
        self.output = []
        self.errors = ''

    def make_tokens(self):

        return self.output, self.errors

    def make_operand(self):
        if self.p[self.i] in digits + '+-':
            token = self.make_num()
            self.output.append(f'TT_imm:{token}')

        elif self.p[self.i] in charset:
            token = self.make_word()
            self.output.append(f'TT_opc:{token}')

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.i += 1
            if prefix in 'rR$':  # register
                if self.p[self.i] not in '+-':
                    self.output.append(f'op_reg:{self.make_num()}')
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix in 'mM#':  # memory
                if self.p[self.i] not in '+-':
                    self.output.append(f'op_reg:{self.make_num()}')
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.output.append(f'op_por{self.make_num()}')
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.output.append(f'op_por{name}')
                    else:
                        self.errors += unk_port.format(self.line_nr)

            elif prefix == '~':  # relative
                self.output.append(f'op_rel:{self.make_num()}')

            elif prefix == '.':  # label
                self.output.append(f'op_lab:{self.make_word()}')

            elif prefix == '@':  # macro
                self.output.append(f'op_mac:{self.make_word()}')

            elif prefix == "'":  # character
                self.output.append('op_cha:' + self.make_str("'"))

            elif prefix == '"':
                self.output.append('op_str:' + self.make_str('"'))

            else:  # unknown operand type
                self.errors += unk_operand.format(self.line_nr)

    def make_str(self, char):
        word = ''
        while self.has_next() and self.p[self.i] != char and self.p[self.i] != '\n':
            word += self.p[self.i]

        if self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append('newLine')
        return word

    def make_word(self):
        word = self.p[self.i]
        self.i += 1
        while self.has_next(0) and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in charset:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                word += self.p[self.i]
            self.i += 1

        if self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append('newLine')
        return word

    def make_num(self):
        num = self.p[self.i]
        self.i += 1
        while self.has_next(0) and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in digits + bases:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                num += self.p[self.i]
            self.i += 1

        if self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append('newLine')
        return int(num, 0)

    def multi_line_comment(self):
        while self.has_next(0) and self.has_next(1) and self.p[self.i] != '*' and self.p[self.i + 1] != '/':
            if self.p[self.i] == '\n':
                self.line_nr += 1
            self.i += 1
        self.i += 1

    def inline_comment(self):
        while self.p[self.i] != '\n':
            self.i += 1
        self.i += 1
        self.line_nr += 1

    def has_next(self, i=0):
        return self.i + i < len(self.p)


if __name__ == "__main__":
    main()
