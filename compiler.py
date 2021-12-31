
# CONSTANTS
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
digits = '1234567890'
bases = 'oOxXbB'
symbols = {
    '(': 'sym_lpa',
    ')': 'sym_rpa',
    '[': 'sym_lbr',
    ']': 'sym_rbr',
}
opcodes = {  # not done

}
port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}

# ERRORS
illegal_char = "Illegal Char '{}' at line {}\n"
invalid_char = "Invalid Character {} at line {}\n"
unk_port = "Unknown port name '{}' at line {}\n"
miss_pair = "Missing closing quote {} at line {}\n"

# TOKENS
t_newLine = 'nln'
t_word = 'wrd:'
t_imm = 'opr_imm:'
t_reg = 'opr_reg:'
t_mem = 'opr_mem:'
t_port = 'opr_por:'
t_relative = 'opr_rel:'
t_label = 'opr_lab:'
t_macro = 'opr_mac:'
t_char = 'opr_cha:'
t_string = 'opr_str:'


def main():
    tok = Lexer(r'''-0X45 dkfh /*267*/40 $4 #50 //comented
.yeet @ohboi  'a' "ah" ~-5 %ASCII''')
    tok.make_tokens()

    if tok.errors != '':
        print(tok.errors)
        return
    print(tok.output)

    # parse

    return


class Lexer:
    def __init__(self, program):
        self.p = program
        self.line_nr = 0
        self.i = 0
        self.output = []
        self.errors = ''

    def make_tokens(self):
        while self.has_next():
            while self.has_next() and self.p[self.i] in ' ,\t':  # ignore commas and indentation
                self.i += 1
            if self.p[self.i] == '\n':  # change line
                self.i += 1
                self.line_nr += 1

            if self.p[self.i] == '/':
                self.i += 1
                if self.has_next() and self.p[self.i] == '/':  # inline comment
                    self.inline_comment()
                elif self.has_next() and self.p[self.i] == '*':
                    self.multi_line_comment()
                else:  # you got your hopes high but it was just an illegal char :/
                    self.errors += illegal_char.format('/', self.line_nr)

            elif self.p[self.i] in symbols:
                self.output.append(symbols[self.p[self.i]])
                self.i += 1
            else:
                self.make_operand()

        return self.output, self.errors

    def make_operand(self):
        if self.p[self.i] in digits + '+-':  # immediate value
            self.output.append(t_imm + str(self.make_num()))

        elif self.p[self.i] in charset:  # opcode
            self.output.append(t_word + self.make_word())

        else:  # format: op_type:<data>
            prefix = self.p[self.i]
            self.i += 1
            if prefix in 'rR$':  # register
                if self.p[self.i] not in '+-':
                    self.output.append(t_reg + 'R' + str(self.make_num()))
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix in 'mM#':  # memory
                if self.p[self.i] not in '+-':
                    self.output.append(t_mem + 'M' + str(self.make_num()))
                else:
                    self.errors += illegal_char.format(self.p[self.i], self.line_nr)

            elif prefix == '%':  # port
                if self.p[self.i] in digits:
                    self.output.append(t_port + '%' + str(self.make_num()))
                else:
                    name = self.make_word()
                    if name in port_names:
                        self.output.append(t_port + '%' + name)
                    else:
                        self.errors += unk_port.format(name, self.line_nr)

            elif prefix == '~':  # relative
                self.output.append(t_relative + prefix + str(self.make_num()))

            elif prefix == '.':  # label
                self.output.append(t_label + prefix + self.make_word())

            elif prefix == '@':  # macro
                self.output.append(t_macro + prefix + self.make_word())

            elif prefix == "'":  # character
                char = self.make_str("'")
                if char == invalid_char:
                    pass
                elif len(char) == 3:  # char = "'<char>'"
                    self.output.append(t_char + char)
                else:
                    self.errors += invalid_char.format(char, self.line_nr)

            elif prefix == '"':
                self.output.append(t_string + self.make_str('"'))

            # elif prefix == '':
            #    self.output.append(f'op_')

            else:  # unknown symbol
                self.errors += illegal_char.format(self.p[self.i-1], self.line_nr)

    def make_str(self, char):
        word = char
        while self.has_next() and self.p[self.i] != char and self.p[self.i] != '\n':
            word += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append(t_newLine)
            self.errors += miss_pair.format(char, self.line_nr)
            return invalid_char
        else:
            word += self.p[self.i]
            self.i += 1
            return word

    def make_word(self):
        word = self.p[self.i]
        self.i += 1
        while self.has_next() and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in charset:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                word += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append(t_newLine)
        return word

    def make_num(self):
        num = self.p[self.i]
        self.i += 1
        while self.has_next() and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in digits + bases:
                self.errors += illegal_char.format(self.p[self.i], self.line_nr)
            else:
                num += self.p[self.i]
            self.i += 1

        if self.has_next() and self.p[self.i] == '\n':
            self.line_nr += 1
            self.output.append(t_newLine)
        return int(num, 0)

    def multi_line_comment(self):
        while self.has_next(1) and (self.p[self.i] != '*' or self.p[self.i + 1] != '/'):
            if self.p[self.i] == '\n':
                self.line_nr += 1
            self.i += 1
        self.i += 2

    def inline_comment(self):
        while self.has_next() and self.p[self.i] != '\n':
            self.i += 1
        self.i += 1
        self.line_nr += 1

    def has_next(self, i=0):
        return self.i + i < len(self.p)


if __name__ == "__main__":
    main()
