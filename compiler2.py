from timeit import default_timer as timer
from math import gcd

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

branch_instructions = {'JMP', 'BGE', 'BRE', 'BNE', 'BRL', 'BRG', 'BLE', 'BZR', 'BNZ', 'BRN', 'BRP', 'BEV', 'BOD', 'CAL',
                       'BRC', 'BNC'}

relative_accepting_instructions = {'JMP', 'BGE', 'BRE', 'BNE', 'BRL', 'BRG', 'BLE', 'BZR', 'BNZ', 'BRN', 'BRP', 'BEV',
                                   'BOD', 'CAL', 'BRC', 'BNC', 'PSH'}

port_names = {'CPUBUS', 'TEXT', 'NUMB', 'SUPPORTED', 'SPECIAL', 'PROFILE', 'X', 'Y', 'COLOR', 'BUFFER', 'G-SPECIAL',
              'ASCII', 'CHAR5', 'CHAR6', 'ASCII7', 'UTF8', 'UTF16', 'UTF32', 'T-SPECIAL', 'INT', 'UINT', 'BIN', 'HEX',
              'FLOAT', 'FIXED', 'N-SPECIAL', 'ADDR', 'BUS', 'PAGE', 'S-SPECIAL', 'RNG', 'NOTE', 'INSTR', 'NLEG', 'WAIT',
              'NADDR', 'DATA', 'M-SPECIAL', 'UD1', 'UD2', 'UD3', 'UD4', 'UD5', 'UD6', 'UD7', 'UD8', 'UD9', 'UD10',
              'UD11', 'UD12', 'UD13', 'UD14', 'UD15', 'UD16'}


def compiler(source):
    start = timer()
    # setup on the program
    source = remove_comments(source)  # removes comments inline or multi line
    source = source.replace(',', '')  # removes commas from the program to maximise compatibility with old programs
    lines = source.split('\n')
    instructions = []
    errors = '```diff\n'
    print('\nCompiling...')

    # setup on library
    lib_code = 'JMP .endFile\n'
    headers = set()  # 'bits', 'minreg', 'minheap', 'run', 'minstack'
    bits_head = '>= 8'

    imported_libraries = set()
    called_lib_functions = set()

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

    labels, errors = label_recogniser(lines)

    ends = end_recogniser(lines)

    for line_nr, line in enumerate(lines):
        if line == '':
            continue

        elif line.startswith(' '):
            line = remove_indent_spaces(line)

        # # # # # # # # # # # # # # # Labels # # # # # # # # # # # # # # #

        if line.startswith('.'):
            instructions.append(line)
            continue

        # # # # # # # # # # # # # # # Instructions # # # # # # # # # # # # # # #

        # big work on instructions starts here :/
        parts = line.split(' ', 1)  # dividing instruction into opcode and operands
        opcode = parts[0]
        operands_str = parts[1]
        operand_count = opcode_op_count(opcode)  # return num of operands of instruction, or YEET if URCLpp/Header/Error
        operands = []

        # # # # # # # # # # # # # # # Library function Calls # # # # # # # # # # # # # # #

        if '(' in operands_str or ')' in operands_str:  # this char is only used in lib calls so it must be func/Error
            if opcode != 'LCAL':  # there is no other instruction that uses parenthesis so it must be an Error
                print(CRED + "Illegal Char Error: '(' used at line " + str(line_nr) + CEND)
                errors += f"-Illegal Char Error: '(' used at line {str(line_nr)}\n"

            if operands_str.count('(') != 1 or operands_str.count(')') != 1:  # only 1 pair of parenthesis allowed
                print(CRED + "Syntax Error: Faulty function Call at line " + str(line_nr) + CEND)
                errors += f"-Syntax Error: Faulty function Call at line {str(line_nr)}\n"
                break

            args_str = operands_str[(operands_str.index('(') + 1):operands_str.index(')')]
            args = args_str.split(' ')
            for arg in args:
                if arg[0] == '@':
                    if arg in macros:
                        arg = macros[arg]
                        operand_type = operand_type_of(arg)

                        if operand_type not in {'imm', 'label', 'mem', 'char', 'port', 'rel'}:
                            print(CRED + "Syntax Error: Invalid macro type passed at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Invalid macro type passed at line {str(line_nr)}\n"
                    else:
                        print(CRED + "Syntax Error: Undefined macro used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Undefined macro used at line {str(line_nr)}\n"

            operands.append(operands_str[0:operands_str.index('(')])
            operands.append(args_str)

        # # # # # # # # # # # # # # # Multiword on the operands # # # # # # # # # # # # # # #

        elif '[' in operands_str or ']' in operands_str:
            if opcode not in multiword_instructions:
                if '[' in operands_str and ']' in operands_str:
                    print(CRED + "Syntax Error: The instruction '" + opcode + "' doesnt support multiword, at line " +
                          str(line_nr) + CEND)
                    errors += f"-Illegal Char Error: '[' or ']' used at line {str(line_nr)}\n"
                else:
                    print(CRED + "Illegal Char Error: '[' or ']' used at line " + str(line_nr) + CEND)
                    errors += f"-Illegal Char Error: '[' or ']' used at line {str(line_nr)}\n"

            args_str = operands_str[(operands_str.index('[') + 1):operands_str.index(']')]

            for arg in args_str.split(' '):
                if arg[0] == '@':
                    if arg in macros:
                        arg = macros[arg]
                        operand_type = operand_type_of(arg)

                        if operand_type not in {'imm', 'label', 'reg'}:
                            print(CRED + "Syntax Error: Invalid macro type used at line " + str(line_nr) + CEND)
                            errors += f"-Syntax Error: Invalid macro type used at line {str(line_nr)}\n"

                    else:
                        print(CRED + "Syntax Error: Undefined macro used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Undefined macro used at line {str(line_nr)}\n"

            operands_str = operands_str.replace(args_str, '')
            if operands_str[0] == ' ':  # then multiword is not the first operand
                if ' ' in operands_str[1:]:  # it has 2 non multiword operands
                    first_2_operands = operands_str[1:].split(' ')
                    assert len(first_2_operands) >= 2
                    operands.append(first_2_operands[0])
                    operands.append(args_str)
                    operands.append(first_2_operands[1])
                else:  # 1 op and a multiword op
                    operands.append(operands_str[1:])
                    operands.append(args_str)

            else:  # multiword is the first operand
                operands.append(args_str)
                if ' ' in operands_str[1:]:
                    first_2_operands = operands_str[1:].split(' ', 2)
                    assert len(first_2_operands) >= 2
                    operands.extend(first_2_operands)
                else:
                    operands.append(operands_str[1:])

        # # # # # # # # # # # # # # # String as a multiword operand on DW # # # # # # # # # # # # # # #

        elif '"' in operands_str:
            if opcode != 'DW' and operands_str.count('"') == 2:  # strings must be in the form of a multiword DW
                print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                      CEND)
                errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

            elif operands_str.count('"') != 2:
                print(CRED + "Illegal Char Error: '\"' used at line " + str(line_nr) + CEND)
                errors += f"-Illegal Char Error: '\"' used at line {str(line_nr)}\n"

            else:
                string_ = operands_str[operands_str.index('"') + 1:operands_str.rindex('"')]
                split_string = list(string_)
                final_operand = '['
                for char in split_string:
                    final_operand += f"'{char}' "

                final_operand = final_operand[:-1] + ']'
                instructions.append('DW ' + final_operand)
                continue

        # # # # # # # # # # # # # # # Operand prefixes # # # # # # # # # # # # # # #

        else:
            operands = operands_str.split(' ')

        valid_operands = []
        # for some reason, arg is already defined here wtf
        for arg in operands:
            operand_type = operand_type_of(arg)

            if opcode == 'LCAL' or opcode == 'IMPORT':  # LCAL has its operands sorted already
                valid_operands.append(arg)

            elif operand_type == 'ERROR':  # its not a valid operand
                print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

            elif operand_type in {'imm', 'reg'}:
                valid_operands.append(arg)

            elif operand_type == 'mem':
                if opcode in memory_instructions:
                    valid_operands.append(arg)
                else:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

            elif operand_type == 'label':
                arg = arg[1:]

                if arg in labels:
                    valid_operands.append(arg)
                else:
                    print(CRED + "Syntax Error: Unknown label used at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Unknown label used at line {str(line_nr)}\n"

            elif operand_type == 'rel':
                if opcode in relative_accepting_instructions:
                    valid_operands.append(arg)
                else:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

            elif operand_type == 'port':
                if opcode in {'IN', 'OUT'}:
                    if operand_type[1:].isnumeric():
                        valid_operands.append(arg)

                    elif operand_type[1:] in port_names:
                        valid_operands.append(arg)
                    else:
                        print(CRED + "Syntax Error: Unknown Port name used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown Port name used at line {str(line_nr)}\n"
                else:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '" + opcode + "' used at line {str(line)}\n"

            elif operand_type == 'char':
                if arg[1:].index("'") == 2:  # special chars like \n or \t or error
                    if arg[1:3] in {'\\n', '\\t', '\\r', '\\b', '\\v', '\\0'}:
                        valid_operands.append(arg)
                    else:
                        print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

                elif len(arg) == 3 and arg[2] == "'":  # normal char
                    valid_operands.append(arg)
                else:
                    print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

            elif operand_type == 'cnd':
                if opcode in conditional_instructions:
                    valid_operands.append(arg)
                else:
                    print(CRED + "Syntax Error: Wrong operand type for  '" + opcode + "' used at line " + str(line_nr)
                          + CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

            elif operand_type == 'macro':
                if opcode == '@define':
                    if operands.index(arg) == 1:  # its declaring a macro based on another macro, and that is a no :P
                        print(CRED + "Syntax Error: Wrong operand type for second operand in '" + opcode +
                              "' used at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Wrong operand type for second operand in '{opcode}' used " \
                                  f"at line {str(line_nr)}\n"
                    else:
                        valid_operands.append(arg)

                elif arg in macros:
                    arg = macros[arg]
                    operand_type = operand_type_of(arg)
                    args_str = macro_operand_valid(operand_type, opcode, line_nr)

                    if args_str == '':
                        valid_operands.append(arg)
                    else:
                        errors += args_str
                        break

                else:
                    print(CRED + "Syntax Error: Undefined macro used at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Undefined macro used at line {str(line_nr)}\n"

        operands = valid_operands

        # # # # # # # # # # # # # # # First Operand type checks # # # # # # # # # # # # # # #

        if operand_count != 'ERROR':  # then its a main URCL instruction
            destination_operand_type = operand_type_of(operands[0])

            if destination_operand_type not in {'reg', 'port', 'rel', 'label', 'mem'}:  # operand 1 must be address/reg
                if destination_operand_type == 'imm' and opcode in memory_instructions:
                    print(CRED + "Warning: Immediate values should NOT be used as addresses in memory instructions at "
                                 "line " + str(line_nr) + CEND)
                    errors += f"-Warning: Immediate values should NOT be used as addresses in memory instructions at " \
                              f"line {str(line_nr)}\n"

                elif destination_operand_type == 'imm' and opcode == 'PSH':  # this is the only exception to the rule
                    pass

                else:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"
                    break

            else:
                if opcode in {'STR', 'LSTR', 'CPY'} and destination_operand_type not in {'mem', 'reg'}:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

                elif opcode in branch_instructions and destination_operand_type not in {'rel', 'reg', 'label', 'imm'}:
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

                elif opcode == 'OUT' and destination_operand_type != 'port':
                    print(CRED + "Syntax Error: Wrong operand type for '" + opcode + "' used at line " + str(line_nr) +
                          CEND)
                    errors += f"-Syntax Error: Wrong operand type for '{opcode}' used at line {str(line_nr)}\n"

        # # # # # # # # # # # # # # # Opcodes # # # # # # # # # # # # # # #

        if operand_count == 'ERROR':  # can be an Error, header or an URCLpp exclusive instruction
            operand_count = new_opcode_op_count(opcode)

            if operand_count == 'ERROR':  # its not an URCLpp instruction neither, so its either an error or header
                operand_count = check_headers(opcode)

                if operand_count == 'ERROR':  # its not an header neither, meaning its an error
                    print(CRED + "Syntax Error: Unknown instruction at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Unknown instruction at line {str(line_nr)}\n"

                # # # # # # # # # # # # # # # Headers # # # # # # # # # # # # # # #

                else:
                    if operand_count != len(operands):
                        print(CRED + "Syntax Error: Wrong number of operands in Header at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Wrong number of operands in Header at line {str(line_nr)}\n"
                    else:

                        if opcode == 'BITS':
                            if 'bits' in headers:
                                print(CRED + "Syntax Error: More than 1 'BITS' header at line " + str(line_nr) + CEND)
                                errors += f"-Syntax Error: More than 1 'BITS' header at line {str(line_nr)}\n"

                            else:
                                headers.add('bits')
                                macros['@BITS'] = operands[1]
                                macros['@MSB'] = str(-(2 ** (int(operands[1]) - 1)))
                                macros['@SMSB'] = str(2 ** (int(operands[1]) - 2))
                                macros['@SMAX'] = str((2 ** (int(operands[1]) - 1)) - 1)
                                macros['@UHALF'] = str(-(2 ** (int(operands[1]) // 2)))
                                macros['@LHALF'] = str((2 ** (int(operands[1]) // 2)) - 1)
                                bits_head = operands_str

                        elif opcode == 'MINREG':
                            if 'minreg' in headers:
                                print(CRED + "Syntax Error: More than 1 'MINREG' header at line " + str(line_nr) + CEND)
                                errors += f"-Syntax Error: More than 1 'MINREG' header at line {str(line_nr)}\n"

                            else:
                                headers.add('minreg')
                                macros['@MINREG'] = operands[0]

                        elif opcode == 'MINHEAP':
                            if 'minheap' in headers:
                                print(CRED + "Syntax Error: More than 1 'MINHEAP' header at line " + str(line_nr) +
                                      CEND)
                                errors += f"-Syntax Error: More than 1 'MINHEAP' header at line {str(line_nr)}\n"

                            else:
                                headers.add('minheap')
                                macros['@MINHEAP'] = operands[0]

                        elif opcode == 'RUN':
                            if 'run' in headers:
                                print(CRED + "Syntax Error: More than 1 'RUN' header at line " + str(line_nr) + CEND)
                                errors += f"-Syntax Error: More than 1 'RUN' header at line {str(line_nr)}\n"

                            else:
                                headers.add('run')
                                macros['@RUN'] = operands[0]

                        elif opcode == 'MINSTACK':
                            if 'minstack' in headers:
                                print(CRED + "Syntax Error: More than 1 'MINSTACK' header at line " + str(line_nr) +
                                      CEND)
                                errors += f"-Syntax Error: More than 1 'MINSTACK' header at line {str(line_nr)}\n"

                            else:
                                headers.add('minstack')
                                macros['@MINSTACK'] = operands[0]

                        elif opcode == 'IMPORT':
                            lib_name = operands[0]
                            if not os.path.isdir(script_dir + r'/libraries/' + lib_name):
                                print(CRED + "Syntax Error: Unknown library at line " + str(line_nr) + CEND)
                                errors += f"-Syntax Error: Unknown library at line {str(line_nr)}\n"

                            elif lib_name in imported_libraries:
                                print(CRED + "Warning: Library already imported in previous statement at line " +
                                      str(line_nr) + CEND)
                                errors += f"-Warning: Library already imported in previous statement at line " \
                                          f"{str(line_nr)}\n"
                            else:
                                imported_libraries.add(lib_name)

            # # # # # # # # # # # # # # # URCLpp instructions # # # # # # # # # # # # # # #

            else:  # its a URCLpp exclusive instruction

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
                    macros[operands[0]] = operands[1]

                # # # # # # # # # # # # # # # Library Call # # # # # # # # # # # # # # #

                elif opcode == 'LCAL':
                    if '(' not in operands_str:
                        print(CRED + "Syntax Error: Faulty library call at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Faulty library call at line {str(line_nr)}\n"
                        break

                    lib = operands[0]
                    lib = lib.replace('.', '/')
                    lib_name = lib.split('/', 1)[0]
                    rel_path = r"/libraries/" + lib + '.urcl'
                    abs_file_path = script_dir + rel_path
                    if lib_name not in imported_libraries:
                        print(CRED + "Syntax Error: Library not imported at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Library not imported at line {str(line_nr)}\n"

                    if os.path.isfile(abs_file_path):
                        lib_output = 0  # ignore: just so IDE doesnt trigger the var referenced before assignment error
                        if lib not in called_lib_functions:
                            called_lib_functions.add(lib)

                            lib_output = lib_importer(abs_file_path, [bits_head, macros['@MINREG']], operands[1], lib)
                            lib_code += lib_output[0] + '\n'  # having an empty line for readability

                        instructions.append('\n' + lib_helper(operands[1], lib_output[1], lib, lib_output[2]))

                    else:
                        print(CRED + "Syntax Error: Unknown library at line " + str(line_nr) + CEND)
                        errors += f"-Syntax Error: Unknown library at line {str(line_nr)}\n"

        # # # # # # # # # # # # # # # Main URCL instruction # # # # # # # # # # # # # # #

        else:  # its a normal instruction
            if operand_count != len(operands):  # either wrong number of operands or use smart typing
                if operand_count - 1 == len(operands):  # smart typing it is
                    instructions.append(opcode + ' ' + str(operands[0]) + ' ' + (' '.join(operands)))
                else:
                    print(CRED + "Syntax Error: Wrong number of operands at line " + str(line_nr) + CEND)
                    errors += f"-Syntax Error: Wrong number of operands at line {str(line_nr)}\n"
            else:  # normal instruction here
                instructions.append(opcode + ' ' + (' '.join(operands)))

    end = timer()
    print(f'Operation Completed in {latency(start, end)}ms!')

    final_program = ''
    for line in instructions:
        final_program += line + '\n'

    if len(called_lib_functions) != 0:
        lib_code += '.endFile'
        final_program += lib_code

    if errors != "```diff\n":
        return errors + final_program

    return final_program


# # # # # # # # # # # # # # # Helper Functions below # # # # # # # # # # # # # # #

def get_input():
    # get sample program to debug in text, open in read text mode
    with open('debug_test.urcl', mode='r') as f:
        return f.read()


def remove_indent_spaces(line):
    i = 0
    while line[i] == ' ':
        if i < len(line):
            i += 1
    return line[i:]


def remove_comments(source):  # removes all inline comments and multiline comments from the program
    i = 0
    output = ''
    commented = False
    while i < len(source):
        if commented:
            try:
                if source[i] == '*' and source[i + 1] == '/':
                    i += 2
                    commented = False
            except IndexError:
                pass
        else:
            try:
                if source[i] == '/' and source[i + 1] == '*':
                    i += 2
                    commented = True
                    continue
                else:
                    if source[i] == '/' and source[i + 1] == '/':
                        i += 2
                        while source[i] != '\n':
                            i += 1
            except IndexError:
                pass
            output += source[i]

        i += 1
    return output


def opcode_op_count(opcode):  # checks if the opcode is correct and returns the number of operands expected
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
        # Directives
        'DW': 1
    }
    try:
        operand_count = operands[opcode]
    except KeyError:
        operand_count = 'ERROR'

    return operand_count


def new_opcode_op_count(opcode):
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
        operand_count = operands[opcode]
    except KeyError:
        operand_count = 'ERROR'

    return operand_count


def check_headers(header_name):
    header = {
        'IMPORT': 1,
        'BITS': 2,
        'MINREG': 1,
        'MINHEAP': 1,
        'RUN': 1,
        'MINSTACK': 1,
    }
    try:
        operant_count = header[header_name]
    except KeyError:
        operant_count = 'ERROR'

    return operant_count


def operand_type_of(operand):
    if operand.isnumeric():  # then its an IMM
        return 'imm'

    elif operand == 'SP':  # sp is a valid operand
        return 'reg'

    else:
        prefix = operand[0]
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
            "'": 'char',
            '=': 'cnd',
            '!': 'cnd',
            '<': 'cnd',
            '>': 'cnd',
            '(': 'arg',  # wont be used but its here anyways
            '[': 'multi',
            '"': 'string',
        }
        try:
            return op_type[prefix]
        except KeyError:
            return 'ERROR'


def label_recogniser(lines):
    labels = set()
    errors = ''

    for line_nr, line in enumerate(lines):
        if line.startswith('.'):
            i = 1
            while i < len(line):  # cannot contain illegal chars
                if line[i] not in allowed_chars:
                    print(CRED + "Illegal Char Error: '" + line[i] + "' used at line " + str(line_nr) + CEND)
                    errors += f"-Illegal Char Error: '{line[i]}' used at line {str(line_nr)}\n"
                i += 1
            if line in labels:  # cant have duplicates
                print(CRED + "Syntax Error: Duplicate label used at line " + str(line_nr) + CEND)
                errors += f"-Syntax Error: Duplicate label used at line {str(line_nr)}\n"
            else:  # all went well here :D
                labels.add(line)
    return [labels, errors]


def end_recogniser(lines):
    ends = []
    for line_nr, line in enumerate(lines):
        if line == 'END':
            ends.append(line_nr)
    return ends


def macro_operand_valid(op_type, opcode, line_nr):
    errors = ''
    if op_type == 'ERROR':  # its not a valid operand
        print(CRED + "Syntax Error: Unknown operand type used at line " + str(line_nr) + CEND)
        errors += f"-Syntax Error: Unknown operand type used at line {str(line_nr)}\n"

    elif op_type == 'mem':
        if opcode not in memory_instructions:
            print(CRED + "Syntax Error: Wrong macro type for  '" + opcode + "' used at line " + str(line_nr) + CEND)
            errors += f"-Syntax Error: Wrong macro type for '{opcode}' used at line {str(line_nr)}\n"

    elif op_type == 'rel':
        if opcode not in relative_accepting_instructions:
            print(CRED + "Syntax Error: Wrong macro type for  '" + opcode + "' used at line " + str(line_nr) + CEND)
            errors += f"-Syntax Error: Wrong macro type for '{opcode}' used at line {str(line_nr)}\n"

    elif op_type == 'port':
        if opcode not in {'IN', 'OUT'}:
            print(CRED + "Syntax Error: Wrong macro type for '" + opcode + "' used at line " + str(line_nr) + CEND)
            errors += f"-Syntax Error: Wrong macro type for '" + opcode + "' used at line {str(line)}\n"

    elif op_type == 'cnd':
        if opcode not in conditional_instructions:
            print(CRED + "Syntax Error: Wrong macro type for '" + opcode + "' used at line " + str(line_nr) + CEND)
            errors += f"-Syntax Error: Wrong macro type for '" + opcode + "' used at line {str(line)}\n"

    elif op_type == 'mutli':
        pass

    else:  # if op_type in {'imm', 'reg', 'char'}:
        pass

    return errors


def latency(start, end):
    total_time = round((end - start)*1000)
    if total_time < 1:
        return '~0'
    else:
        return f'~{total_time}'


def lib_importer(abs_file_path, headers, args, lib_name):
    args = args[1:-1]
    args = args.split(' ')
    with open(abs_file_path) as f:
        lib_function = f.read()
        program = remove_comments(lib_function)
        program = program.replace(',', '')
        lines = program.split('\n')
        errors = ''

        lib_headers = [False, False, False, False]
        headers_done = False
        output = '.' + lib_name + '\n'
        regs_needed = 0
        output_regs = 0

        for line_num, line in enumerate(lines):
            part = line.split(' ', 1)
            if line == '':
                continue
            try:
                operands_str = part[1]
            except IndexError:
                pass
            operand = operands_str.split(' ')

            if headers_done:
                output += line + '\n'
            else:
                if line.startswith('BITS'):
                    lib_headers[0] = True
                    if bits_compatibility(operands_str, headers[0]):
                        print(CRED + "Compatibility Error: Incompatible library function" + CEND)
                        errors += f"-Compatibility Error: Incompatible library function"
                        return errors

                elif line.startswith('OPS'):
                    lib_headers[1] = True
                    if int(operand[0]) < len(args):
                        print(CRED + "Type Error: Too many arguments given in library call at line " +
                              str(line_num) + CEND)
                        errors += f"-Type Error: Too many arguments given in library call at line " \
                                  f"{str(line_num)}\n"
                        return errors

                    elif int(operand[0]) > len(args):
                        print(CRED + "Type Error: Missing argument in library call at line " + str(line_num) +
                              CEND)
                        errors += f"-Type Error: Missing argument in library call at line {str(line_num)}\n"
                        return errors

                elif line.startswith('REG'):
                    lib_headers[2] = True
                    regs_needed = int(operand[0])
                    if int(headers[1]) < int(operand[0]):
                        print(CRED + "Type Error: Not enough registers available for library function" + CEND)
                        errors += f"-Type Error: Not enough registers available for library function"
                        return errors

                elif line.startswith('OUTS'):
                    lib_headers[3] = True
                    output_regs = operand[0]

                headers_done = lib_headers[0] and lib_headers[1] and lib_headers[2]

    return [output, regs_needed, output_regs]


def lib_helper(args, regs, lib_name, output_regs_num):
    args = args.split(' ')
    args_passed = ''
    saved_regs = ''
    restored_regs = ''
    moved_regs = ''
    destination_regs = set()

    calling_instruction = 'CAL .' + lib_name + '\n'

    for num in range(int(output_regs_num) + 1):
        print(args)
        if args[num][0] == 'R' or args[num][0] == '$':
            moved_regs += f'MOV {args[num]} R{num + 1}\n'
            destination_regs.add(num)

    for reg in range(1, regs + 1):
        if reg not in destination_regs:
            saved_regs += f'PSH R{reg}\n'

    args.reverse()
    for num in range(1, int(output_regs_num) + 1):
        args.pop()

    for arg in args:
        args_passed += f'PSH {arg}\n'

    for num in range(regs, 0, -1):
        restored_regs += f'POP R{num}\n'

    return saved_regs + args_passed + calling_instruction + moved_regs + restored_regs


def bits_compatibility(lib_header, header):
    header = header.split(' ', 1)

    for num in range(1, 65):
        if eval(header[1] + ' ' + header[0] + ' ' + str(num) + lib_header):
            return False

    return True


print(compiler(get_input()))
