
# ONLY CHANGE THE ITEMS INSIDE THIS FUNCTION V

func_names = ['func_name']  # to change the

# DON'T CHANGE THE FOLLOWING CODE

from inspect import signature

func_args = {}
for a in func_names:
    func_args[a] = len(signature(eval(a)).parameters)

# BELOW YOU CAN EDIT THIS FILE

# Create your functions here following the example:
"""def func_name(a, b, c):  # note that a, b, c are the operands in the code of your function
    return f'''ADD {a} {b} {c}
RSH {b} {a}'''

# to use your opcodes have them inside curly brackets like this: {a}

"""

# END OF FILE
