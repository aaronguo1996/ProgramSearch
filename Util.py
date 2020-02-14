import numpy as np
import string

MAX_STR_LEN = 36
CHARACTERS = string.printable[:-4]
DELIMITERS = "& , . ? ! @ ( ) [ ] % { } / : ; $ # \" ' -".split(' ') + [" "]
BOUNDARY = ["Start", "End"]
POSITION_K = list(range(-MAX_STR_LEN, MAX_STR_LEN + 1))
INDEX = list(range(-5, 6))


def str_to_np(list_of_str):
    """
    turn a list of strings into a numpy representation
    """
    ret = np.zeros(shape = (len(list_of_str), MAX_STR_LEN))

    for i, str in enumerate(list_of_str):
        for j, c in enumerate(str):
            ret[i][j] = CHARACTERS.index(c) + 1

    return ret
