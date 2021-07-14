
def lexer(self):
    labels = {}
    instructions = []
    self = remove_comments(self)
    lines = self.split('\n')
    line = 0
    for a in lines:
        a = a.split(' ', 1)
        if a[0].startswith('.'):
            labels[a[0]] = line
        else:
            opcode = opcodes(a[0])
            if opcode == 'YEET':
                pass  # check header stuff
            else:
                operand = a[1].split(' ')

        line += 1
    return


def remove_comments(self):
    i = 0
    output = ''
    commented = False
    while i < len(self):
        if commented:
            try:
                if self[i] == '*' and self[i + 1] == '/':
                    i += 2
                    commented = False
            except IndexError:
                pass
        else:
            try:
                if self[i] == '/' and self[i + 1] == '*':
                    i += 2
                    commented = True
                else:
                    if self[i] == '/' and self[i + 1] == '/':
                        i += 2
                        while self[i] != '\n':
                            i += 1
            except IndexError:
                pass
            output += self[i]

        i += 1
    return output


def opcodes(self):
    opcode = {
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        '': '',
        # urcl++ exclusive below
        '': '',
    }
    try:
        output = opcode[self]
    except IndexError:
        output = 'YEET'
    return output
