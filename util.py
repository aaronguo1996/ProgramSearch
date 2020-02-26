import numpy as np
import string
import re

N_EXPRS = 6
MAX_CONST_LEN = 5
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
    'Word'     : r'[A-Za-z]+',
    'Alphanum' : r'[A-Za-z0-9]',
    'PropCase' : r'[A-Z][a-z]*',
    'AllCaps'  : r'[A-Z]',
    'Lower'    : r'[a-z]',
    'Digit'    : r'[0-9]',
    'Char'     : r'.'
}
DELIM_REGEX = {}
for d in DELIMITERS:
    DELIM_REGEX[d] = re.escape(d)
ALL_REGEX = {}
ALL_REGEX.update(DELIM_REGEX)
ALL_REGEX.update(REGEX)

# parameters for encoding
CHAR_EMBED_DIM = 20
KERNEL_SIZE = 5
COLUMN_ENCODING_DIM = 32
ACTION_EMBED_DIM = 32
DENSE_LAYERS = 10
GROWTH_RATE = 128
STR_LEN = 100
H_OUT = 128
N_IO = 4

# parameters for training
BATCH_SIZE = 4000
TRAIN_ITERATIONS = 50000
N_ENVS_PER_ROLLOUT = 2
RL_ITERATIONS = 12000
N_PROCESSES = 8
USE_PARALLEL = True

# model store
SAVE_PATH = 'models/model_'
LOAD_PATH = 'models/model_'
PRINT_FREQ = 2
SAVE_FREQ = 100
TEST_FREQ = 1000

def str_to_np(list_of_str):
    """
    turn a list of strings into a numpy representation
    """
    ret = np.zeros(shape = (len(list_of_str), MAX_STR_LEN))

    for i, str in enumerate(list_of_str):
        for j, c in enumerate(str):
            ret[i][j] = CHARACTERS.index(c) + 1

    return ret

def step_abs(i):
    """
    compute the absolute value for a step
    """
    return i + 1 if i >=0 else -i

def merge_constraints(constraint_list):
    """
    constraints = (regex counts, string length)
    merge a list of constraints into one tuple:
        1) choose the largest count for each regex
        2) choose the largest length
    """

    res_d = {}
    res_l = 0

    for c in constraint_list:
        d, l = c

        # merge the dictionary
        for r in d:
            if r in res_d:
                res_d[r] = max(res_d[r], d[r])
            else:
                res_d[r] = d[r]

        # merge the integer length
        res_l = max(res_l, l)

    return res_d, res_l

def execute_actions(actions, pstate):
    """
    sequentially execute all the actions in the given action list
    """

    for a in actions:
        pstate = a(pstate)
        # print(pstate.committed)

    return pstate
