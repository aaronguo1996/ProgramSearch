import numpy as np
import string
import re

N_EXPRS = 6
MAX_LEN = 5
MAX_STR_LEN = 36
MAX_MASK_LEN = 7
CHARACTERS = string.printable[:-4]
DELIMITERS = '& , . ? ! @ ( ) [ ] % { } / : ; $ # " \' -'.split(' ') + [' ']
BOUNDARY = ['Start', 'End']
CASES = ['Proper', 'AllCaps', 'Lower']
POSITION_K = list(range(-MAX_STR_LEN, MAX_STR_LEN + 1))
INDEX = list(range(-5, 6))
REGEX = {
    'Number'   : r'[0-9]+',
    'Word'     : r'[a-zA-Z0-9_]+',
    'Alphanum' : r'[A-Za-z0-9_]',
    'PropCase' : r'[A-Z][a-z]*',
    'AllCaps'  : r'[A-Z]',
    'Lower'    : r'[a-z]',
    'Digit'    : r'[0-9]',
    'Char'     : r'.'
}
DELIM_REGEX = {}
for d in DELIMITERS:
    DELIM_REGEX[d] = re.escape(d)
ALL_REGEX = REGEX.update(DELIM_REGEX)

def str_to_np(list_of_str):
    """
    turn a list of strings into a numpy representation
    """
    ret = np.zeros(shape = (len(list_of_str), MAX_STR_LEN))

    for i, str in enumerate(list_of_str):
        for j, c in enumerate(str):
            ret[i][j] = CHARACTERS.index(c) + 1

    return ret
