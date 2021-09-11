import os

CRED = '\033[91m'
CGREEN = '\033[32m'
CEND = '\033[0m'
allowed_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'


def compiler(self):
    # setup on the program
    self = remove_comments(self)  # removes comments inline or multi line
    self = self.replace(',', '')  # removes commas from the program to maximise compatibility with old programs
    lines = self.split('\n')
    instructions = []
    errors = ''

    # setup on library
    lib_code = 'JMP .endFile\n'
    header = {
        'bits': False,
        'minreg': False,
        'minheap': False,
        'run': False,
        'minstack': False,
        'import': False
    }

    # other setups
    patterns = {
        '&BITS': 8,
        '&MINREG': 8,
        '&MINHEAP': 16,
        '&MINSTACK': 8,
        '&MSB': -128,
        '&SMSB': 64,
        '&MAX': -1,
        '&SMAX': 127,
        '&UHALF': -16,
        '&LHALF': 15,
    }
    labels = set()
    macros = {}

    for line, a in enumerate(lines):
        if a == '\n':
            break
        elif a.startswith(' '):
            a = remove_indent_spaces(a)

        # # # # # # # # # # # # # # # Labels get sorted here # # # # # # # # # # # # # # #

        if a.startswith('.'):  # check duplicated labels and paste them
            i = 1
            while i < len(a):  # cannot contain illegal chars
                if a[i] not in allowed_chars:
                    print(CRED + "Illegal Char Error: '" + a[i] + "' used at line " + str(line) + CEND)
                    errors += f"-Illegal Char Error: '{a[i]}' used at line {str(line)}\n"
                i += 1
            if a in labels:  # cant have duplicates
                print(CRED + "Syntax Error: Duplicate label used at line " + str(line) + CEND)
                errors += f"-Syntax Error: Duplicate label used at line {str(line)}\n"
            else:  # all went well here :D
                labels.add(a)
                instructions.append(a)

        # # # # # # # # # # # # # # # Instructions # # # # # # # # # # # # # # #

        else:  # big work on instructions starts here :/
            a = a.split(' ', 1)  # dividing instruction into opcode and operands
            opcode = a[0]
            operands = a[1]
            op_num = opcodes(opcode)  # returns the n of operands the instruction needs, or YEET if URCLpp/Header/Error
            operand = []

            # # # # # # # # # # # # # # # Library function Calls # # # # # # # # # # # # # # #

            if '(' in operand or ')' in operand:  # this char is only used in library calls so it must be function/Error
                if opcode != 'LCAL':  # there is no other instruction that uses parenthesis so it must be an Error
                    print(CRED + "Illegal Char Error: '(' used at line " + str(line) + CEND)
                    errors += f"-Illegal Char Error: '(' used at line {str(line)}\n"

                if operands.count('(') != 1 or operands.count(')') != 1:  # only 1 pair of parenthesis allowed
                    print(CRED + "Syntax Error: Faulty function Call at line " + str(line) + CEND)
                    errors += f"-Syntax Error: Faulty function Call at line {str(line)}\n"
                lib = operands[0:operands.index('(')]
                lib = lib.replace('.', '/')
                lib_name = lib.split('/')[0]
                script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
                rel_path = r"libraries/" + lib
                abs_file_path = script_dir + rel_path

                if os.path.isfile(abs_file_path):
                    with open(abs_file_path) as f:
                        lib_function = f.read()

                else:
                    if os.path.isdir(script_dir + r'libs/' + lib_name):
                        print(CRED + "Syntax Error: Unknown library function at line " + str(line) + CEND)
                        errors += f"-Syntax Error: Unknown library function at line {str(line)}\n"
                    else:
                        print(CRED + "Syntax Error: Unknown library at line " + str(line) + CEND)
                        errors += f"-Syntax Error: Unknown library at line {str(line)}\n"

            # # # # # # # # # # # # # # # Multiword on the operands # # # # # # # # # # # # # # #

            if '[' in operand or ']' in operand:
                if opcode not in {'LOD', 'LLOD', 'STR', 'LSTR', 'JMP', 'CPY', 'BGE', 'BRE', 'BNE', 'BRL', 'BRG', 'BLE',
                                  'BZR', 'BNZ', 'BRN', 'BRP', 'BEV', 'BOD', 'CAL', 'BRC', 'BNC', 'DW'}:
                    if '[' in operand and ']' in operand:
                        print(CRED + "Syntax Error: The instruction '" + opcode + "' doesnt support multiword, at line "
                              + str(line) + CEND)
                        errors += f"-Illegal Char Error: '[' or ']' used at line {str(line)}\n"
                    else:
                        print(CRED + "Illegal Char Error: '[' or ']' used at line " + str(line) + CEND)
                        errors += f"-Illegal Char Error: '[' or ']' used at line {str(line)}\n"

            # # # # # # # # # # # # # # # Operand prefixes # # # # # # # # # # # # # # #

            # # # # # # # # # # # # # # # Macros # # # # # # # # # # # # # # #

            # # # # # # # # # # # # # # #  # # # # # # # # # # # # # # #

            if op_num == 'YEET':  # can be an Error, header or an URCLpp exclusive instruction
                op_num = new_opcodes(opcode)
                if op_num == 'YEET':  # its not an URCLpp instruction neither, so its either an error or header
                    if opcode in {'BITS', 'MINREG', 'MINHEAP', 'MINSTACK', 'RUN', 'IMPORT'}:  # its an Header

                        # # # # # # # # # # # # # # # Headers # # # # # # # # # # # # # # #

                        pass
                    else:  # its not an header neither, meaning its an error
                        print(CRED + "Syntax Error: Unknown instruction at line " + str(line) + CEND)
                        errors += f"-Syntax Error: Unknown instruction at line {str(line)}\n"
                else:  # its a URCLpp exclusive instruction

                    # # # # # # # # # # # # # # # URCLpp instruction # # # # # # # # # # # # # # #

                    pass
            else:  # its a normal instruction

                # # # # # # # # # # # # # # # Main URCL instruction # # # # # # # # # # # # # # #

                if op_num != len(operand):  # either wrong number of operands or use smart typing
                    if op_num + 1 == len(operand):  # smart typing it is
                        instructions.append(opcode + ' ' + str(operand[0]) + ' ' + (' '.join(operand)))
                    else:
                        print(CRED + "Syntax Error: Wrong number of operands at line " + str(line) + CEND)
                        errors += f"-Syntax Error: Wrong number of operands at line {str(line)}\n"
                else:  # normal instruction here
                    instructions.append(opcode + ' ' + (' '.join(operand)))

    return 'yeet'


# # # # # # # # # # # # # # # Helper Functions below # # # # # # # # # # # # # # #

def get_input():
    return input('Paste here your program:\n')


def remove_indent_spaces(self):
    i = 0
    while self[i] == ' ':
        if i < len(self):
            i += 1
    return self[i:]


def remove_comments(self):  # removes all inline comments and multiline comments from the program
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


def opcodes(self):  # checks if the opcode is correct and returns the number of operands expected
    operands = {
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
        # Directives
        'DW': 1
    }
    try:
        output = operands[self]
    except KeyError:
        output = 'YEET'

    return output


def new_opcodes(self):
    operands = {
        # urcl++ exclusive below
        'LCAL': 2,  # is never used but its here anyways
        '@define': 2,
        'IF': 3,
        'ELIF': 3,
        'ELSE': 3,
        'FOR': 2,
        'WHILE': 3,
        'SWITCH': 1,
        'CASE': 1
    }
    try:
        output = operands[self]
    except KeyError:
        output = 'YEET'

    return output


def lib_helper(self):  # must push and pop the args used and save and restore the registers
    # remove the headers and add some push and poping to save the used registers
    return  # output


print(compiler(get_input()))
