#CONSTANTS
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_1234567890'
digits = '1234567890'
bases = 'oOxXbB'

# ERRORS
illegal_char = "Illegal Char '{}' at line {}\n"


def main():
    test = Lexer('0X45')
    print(test.make_num())
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
        if self.p[self.i] in digits:
            token = self.make_num()

        elif self.p[self.i] in charset:
            token = self.make_word()

        else:
            d = {

            }

    def make_word(self):
        word = self.p[self.i]
        self.i += 1
        while self.has_next(0) and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in charset:
                self.errors += illegal_char.format(self.p[self.i]), self.line_nr
            else:
                word += self.p[self.i]
            self.i += 1
        return word

    def make_num(self):
        num = self.p[self.i]
        self.i += 1
        while self.has_next(0) and self.p[self.i] != ' ' and self.p[self.i] != '\n':
            if self.p[self.i] not in digits + bases:
                self.errors += illegal_char.format(self.p[self.i]), self.line_nr
            else:
                num += self.p[self.i]
            self.i += 1
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

    def has_next(self, i):
        return self.i + i < len(self.p)


if __name__ == "__main__":
    main()
