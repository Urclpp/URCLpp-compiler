def save(*args):
    output = ''
    for a in args:
        output += f'PSH {a}\n'
    return output


def restore(*args):
    output = ''
    for a in reversed(args):
        output += f'POP {a}\n'
    return output

