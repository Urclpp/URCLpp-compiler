import os
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

CRED = '\033[91m'
CGREEN = '\033[32m'
CEND = '\033[0m'
allowed_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
memory_instructions = {'LOD', 'LLOD', 'STR', 'LSTR', 'CPY'}
conditional_instructions = {'IF', 'ELIF', 'WHILE'}

multiword_instructions = {'LOD', 'LLOD', 'STR', 'LSTR', 'JMP', 'CPY', 'BGE', 'BRE', 'BNE', 'BRL', 'BRG', 'BLE', 'BZR',
                          'BNZ', 'BRN', 'BRP', 'BEV', 'BOD', 'CAL', 'BRC', 'BNC', 'DW'}

relative_accepting_instructions = {'JMP', 'BGE', 'BRE', 'BNE', 'BRL', 'BRG', 'BLE', 'BZR', 'BNZ', 'BRN', 'BRP', 'BEV',
                                   'BOD', 'CAL', 'BRC', 'BNC', 'PSH'}

port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}


def compiler(self):
    # setup on the program
    self = remove_comments(self)  # removes comments inline or multi line
    self = self.replace(',', '')  # removes commas from the program to maximise compatibility with old programs
    lines = self.split('\n')
    instructions = []
    errors = ''

    # setup on library
    lib_code = 'JMP .endFile\n'
    headers = set()  # 'bits', 'minreg', 'minheap', 'run', 'minstack'

    imported_libraries = set()

    # other setups
    macros = {
        '@BITS': '8',
        '@MINREG': '8',
        '@MINHEAP': '16',
        '@MINSTACK': '8',
        '@MSB': '-128',
        '@SMSB': '64',
        '@MAX': '-1',
        '@SMAX': '127',
        '@UHALF': '-16',
        '@LHALF': '15',
    }

    temp = label_recogniser(lines)
    labels = temp[0]
    errors += temp[1]

    ends = end_recogniser(lines)

    for line_nr, line in enumerate(lines):
        if line == '\n':
            break
        elif line.startswith(' '):
            line = remove_indent_spaces(line)

        # # # # # # # # # # # # # # # Labels # # # # # # # # # # # # # # #

        if line.startswith('.'):
            instructions.append(line)

        # # # # # # # # # # # # # # # Instructions # # # # # # # # # # # # # # #

        else:  # big work on instructions starts here :/
            opcode, operands_str = line.split(' ', 1)  # dividing instruction into opcode and operands
            operand_count = opcodes(opcode)  # returns the n of operands the instruction needs, or YEET if URCLpp/Header/Error
            operand_list = []

            # # # # # # # # # # # # # # # Library function Calls # # # # # # # # # # # # # # #

            if '(' in operands_str or ')' in operands_str:  # this char is only used in lib calls so it must be function/Error
                if opcode != 'LCAL':  # there is no other instruction that uses parenthesis so it must be an Error
                    print(CRED + "Illegal Char Error: '(' used at line " + str(line_nr) + CEND)
                    errors += f"-Illegal Char Error: '(' used at line {str(line_nr)}\n"

                if operands_str.count('(') != 1 or operands_str.count(')') != 1:  # only 1 pair of parenthesis allowed
                    print(CRED + "Syntax Error: Faulty function Call at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Faulty function Call at line {str(line_nr)}\n"
                    break

            # # # # # # # # # # # # # # # Multiword on the operands # # # # # # # # # # # # # # #

            elif '[' in operands_str or ']' in operands_str:
                if opcode not in multiword_instructions:
                    if '[' in operands_str and ']' in operands_str:
                        print(CRED + "Syntax Error: The instruction '" + opcode + "' doesnt support multiword, at line "
                              + str(line_nr) + CEND)
                        errors += f"-Illegal Char Error: '[' or ']' used at line {str(line_nr)}\n"
                    else:
                        print(CRED + "Illegal Char Error: '[' or ']' used at line " + str(line_nr) + CEND)
                        errors += f"-Illegal Char Error: '[' or ']' used at line {str(line_nr)}\n"

            # # # # # # # # # # # # # # # Operand prefixes # # # # # # # # # # # # # # #

            else:
                operand_list = operands_str.split(' ')

            new_operand_list = []
            for operand in operand_list:
                operand_type = operand_type_of(operand)
                if operand_type == 'ERROR':  # its not a valid operand or its a library name/path
                    if opcode != 'LCAL':
                        print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

                elif operand_type in {'imm', 'reg'}:
                    new_operand_list.append(operand)

                elif operand_type == 'mem':
                    if opcode in memory_instructions:
                        new_operand_list.append(operand)
                    else:
                        print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr)
                              + CEND)
                        errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

                elif operand_type == 'label':
                    operand = operand[1:]

                    if operand in labels:
                        new_operand_list.append(operand)
                    else:
                        print(CRED + "Syntax Error: Unknown label used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown label used at line {str(line_nr)}\n"

                elif operand_type == 'rel':
                    if opcode in relative_accepting_instructions:
                        new_operand_list.append(operand)
                    else:
                        print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                              CEND)
                        errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

                elif operand_type == 'port':
                    if opcode in {'IN', 'OUT'}:
                        if operand_type[1:].isnumeric():
                            new_operand_list.append(operand)

                        elif operand_type[1:] in port_names:
                            new_operand_list.append(operand)
                        else:
                            print(CRED + "Syntax Error: Unknown Port name used at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Unknown Port name used at line {str(line_nr)}\n"
                    else:
                        print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr)
                              + CEND)
                        errors += f"-Syntax Error: Wrong operand type for '" + opcode + "' used at line {str(line)}\n"

                elif operand_type == 'char':
                    if operand[1:].index("'") == 2:  # special chars like \n or \t or error
                        if operand[1:3] in {'\\n', '\\t', '\\r', '\\b', '\\v', '\\0'}:
                            new_operand_list.append(operand)
                        else:
                            print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

                    elif len(operand) == 3 and operand[2] == "'":  # normal char
                        new_operand_list.append(operand)
                    else:
                        print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

                elif operand_type == 'cnd':
                    if opcode in conditional_instructions:
                        new_operand_list.append(operand)
                    else:
                        print(CRED + "Syntax Error: Wrong operand type for  '" + opcode + "' used at line " + str(line_nr)
                              + CEND)
                        errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

                elif operand_type == 'macro':
                    if opcode == '@define':
                        if operand_list.index(operand) == 1:  # its declaring a macro based on another macro, and that is a no :P
                            print(CRED + "Syntax Error: Wrong operand type for second operand in '" + opcode +
                                  "' used at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Wrong operand type for second operand in '{opcode}' used " \
                                      f"at line {str(line_nr)}\n"

                    elif operand[1:] in macros:
                        operand = macros[operand]
                        operand_type = operand_type_of(operand)
                        if operand_type == 'ERROR':  # its not a valid operand
                            print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

                        elif operand_type == 'mem':
                            if opcode in memory_instructions:
                                new_operand_list.append(operand)
                            else:
                                print(CRED + "Syntax Error: Wrong macro type for  '" + opcode + "' used at line " +
                                      str(line_nr) + CEND)
                                errors += f"-Syntax Error: Wrong macro type for '{opcode}' used at line {str(line_nr)}\n"

                        elif operand_type == 'rel':
                            if opcode in relative_accepting_instructions:
                                new_operand_list.append(operand)
                            else:
                                print(
                                    CRED + "Syntax Error: Wrong macro type for '" + opcode + "' used at line " + str(
                                        line_nr) +
                                    CEND)
                                errors += f"-Syntax Error: Wrong macro type for '{opcode}' used at line {str(line_nr)}\n"

                        elif operand_type == 'port':
                            if opcode in {'IN', 'OUT'}:
                                new_operand_list.append(operand)
                            else:
                                print(
                                    CRED + "Syntax Error: Wrong macro type for '" + opcode + "' used at line " + str(
                                        line_nr) + CEND)
                                errors += f"-Syntax Error: Wrong macro type for '" + opcode + "' used at line" \
                                                                                                " {str(line)}\n"

                        elif operand_type == 'cnd':
                            if opcode in conditional_instructions:
                                new_operand_list.append(operand)
                            else:
                                print(CRED + "Syntax Error: Wrong macro type for  '" + opcode + "' used at line " +
                                    str(line_nr) + CEND)
                                errors += f"-Syntax Error: Wrong macro type for '{opcode}' used at line {str(line_nr)}\n"

                        else:  # if operand_type in {'imm', 'reg', 'char'}:
                            new_operand_list.append(operand)

            operand_list = new_operand_list

            # # # # # # # # # # # # # # # Opcodes # # # # # # # # # # # # # # #

            if operand_count == 'ERROR':  # can be an Error, header or an URCLpp exclusive instruction
                operand_count = new_opcodes(opcode)

                if operand_count == 'ERROR':  # its not an URCLpp instruction neither, so its either an error or header
                    operand_count = check_headers(opcode)

                    if operand_count == 'ERROR':  # its not an header neither, meaning its an error
                        print(CRED + "Syntax Error: Unknown instruction at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown instruction at line {str(line_nr)}\n"
                    else:
                        if operand_count != len(operand_list):
                            print(CRED + "Syntax Error: Wrong number of operands at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Wrong number of operands at line {str(line_nr)}\n"
                        else:

                            # # # # # # # # # # # # # # # Headers # # # # # # # # # # # # # # #

                            if opcode == 'BITS':
                                if 'bits' in headers:
                                    print(CRED + "Syntax Error: More than 1 'BITS' header at line " + str(line_nr) + CEND)
                                    errors += f"-Syntax Error: More than 1 'BITS' header at line {str(line_nr)}\n"

                                else:
                                    headers.add('bits')
                                    macros['@BITS'] = operand_list[1]

                            elif opcode == 'MINREG':
                                headers.add('minreg')

                            elif opcode == 'MINHEAP':
                                headers.add('')

                            elif opcode == 'RUN':
                                headers.add('')

                            elif opcode == 'MINSTACK':
                                headers.add('')

                            elif opcode == 'IMPORT':
                                lib_name = operand_list[0]
                                if not os.path.isdir(script_dir + r'libs/' + lib_name):
                                    print(CRED + "Syntax Error: Unknown library at line " + str(line_nr) + CEND)
                                    errors += f"-Syntax Error: Unknown library at line {str(line_nr)}\n"

                else:  # its a URCLpp exclusive instruction

                    # # # # # # # # # # # # # # # URCLpp instructions # # # # # # # # # # # # # # #

                    if opcode == 'END':  # ignore as its not used
                        pass

                    # # # # # # # # # # # # # # # Conditionals # # # # # # # # # # # # # # #

                    elif opcode == 'IF':

                        pass

                    elif opcode == 'ELIF':
                        pass

                    elif opcode == 'ELSE':
                        pass

                    # # # # # # # # # # # # # # # Loops # # # # # # # # # # # # # # #

                    elif opcode == 'FOR':
                        pass

                    elif opcode == 'WHILE':
                        pass

                    # # # # # # # # # # # # # # # Defining Macros # # # # # # # # # # # # # # #

                    if opcode == '@define':

                        pass

                    # # # # # # # # # # # # # # # Library Call # # # # # # # # # # # # # # #

                    if opcode == 'LCAL':
                        lib = operands_str[0:operands_str.index('(')]
                        lib = lib.replace('.', '/')
                        rel_path = r"libraries/" + lib
                        abs_file_path = script_dir + rel_path

                        if os.path.isfile(abs_file_path):
                            with open(abs_file_path) as f:
                                lib_function = f.read()  # work here

                        else:
                            print(CRED + "Syntax Error: Unknown library at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Unknown library at line {str(line_nr)}\n"

            else:  # its a normal instruction

                # # # # # # # # # # # # # # # Main URCL instruction # # # # # # # # # # # # # # #

                if operand_count != len(operand_list):  # either wrong number of operands or use smart typing
                    if operand_count + 1 == len(operand_list):  # smart typing it is
                        instructions.append(opcode + ' ' + str(operand_list[0]) + ' ' + (' '.join(operand_list)))
                    else:
                        print(CRED + "Syntax Error: Wrong number of operands at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Wrong number of operands at line {str(line_nr)}\n"
                else:  # normal instruction here
                    instructions.append(opcode + ' ' + (' '.join(operand_list)))

    return


# # # # # # # # # # # # # # # Helper Functions below # # # # # # # # # # # # # # #

def get_input():
    return """OUT 1 'H'
OUT 1 'e'
OUT 1 'l'
OUT 1 'l'
OUT 1 'o'
OUT 1 ' '
OUT 1 'W'
OUT 1 'o'
OUT 1 'r'
OUT 1 'l'
OUT 1 'd'"""


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
        output = 'ERROR'

    return output


def new_opcodes(self):
    operands = {
        # urcl++ exclusive below
        'LCAL': 2,
        '@define': 2,
        'IF': 3,
        'ELIF': 3,
        'ELSE': 3,
        'FOR': 2,
        'WHILE': 3,
        'SWITCH': 1,
        'CASE': 1,
        'END': 0,
    }
    try:
        output = operands[self]
    except KeyError:
        output = 'ERROR'

    return output


def check_headers(self):
    header = {
        'IMPORT': 2,
        'BITS': 2,
        'MINREG': 1,
        'MINHEAP': 1,
        'RUN': 1,
        'MINSTACK': 1,
    }
    try:
        output = header[self]
    except KeyError:
        output = 'ERROR'

    return output


def operand_type_of(self):
    if self.isnumeric():  # then its an IMM
        return 'imm'
    elif self[0] == "'":  # then its a char
        return 'char'
    else:
        prefix = self[0]
        op_type = {
            'R': 'reg',
            '$': 'reg',
            '#': 'mem',
            'M': 'mem',
            '%': 'port',
            '.': 'label',
            '+': 'imm',
            '-': 'imm',
            '@': 'macro',
            '~': 'rel',
            '=': 'cnd',
            '!': 'cnd',
            '<': 'cnd',
            '>': 'cnd'
        }
        try:
            return op_type[prefix]

        except KeyError:
            return 'ERROR'


def label_recogniser(self):
    labels = set()
    errors = ''

    for line, a in enumerate(self):
        if a.startswith('.'):
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
    return [labels, errors]


def end_recogniser(self):
    ends = []
    for line_nr, line in enumerate(self):
        if line == 'END':
            ends.append(line_nr)
    return ends


def lib_helper(self):  # must push and pop the args used and save and restore the registers
    # remove the headers and add some push and poping to save the used registers
    return  # output


print(compiler(get_input()))
