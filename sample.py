import random
import string
import re
import exrex

import action
import expression
import pregex as pre
from util import *

"""
random sampling of programs
"""

def generate_program():
    """
    generate a random string manipulation program
    """
    return expression.Program.generate()

def insert_regex(in_str, choice_len, regex_constraint):
    """
    insert sampling string for regex if needed
    """
    insert_pos = set(range(choice_len))
    for r in regex_constraint:
        desired_counts = regex_constraint[r]
        actual_counts = len(re.findall(ALL_REGEX[r], ''.join(in_str)))
        num_of_inserts = max(0, desired_counts - actual_counts)

        if len(insert_pos) < num_of_inserts:
            raise IndexError

        indices_of_inserts = set(random.sample(insert_pos,
                                               k=num_of_inserts))

        for i in indices_of_inserts:
            in_str[i] = pre.create(ALL_REGEX[r]).sample()

        # print('Expected', desired_counts, r)
        # print('Actual number is', actual_counts)
        # print('After insert', ''.join(in_str))
        insert_pos -= indices_of_inserts

    str = ''.join(in_str)
    return str[:choice_len] if len(str) > choice_len else str

def generate_valid_input(constraints, max_string_size=MAX_STR_LEN):
    """
    generate a random but valid input string
    """
    min_regex, min_len = constraints
    choice_len = random.randint(min_len, max_string_size)
    random_in = random.choices(CHARACTERS, k=choice_len)
    return insert_regex(random_in, choice_len, min_regex)

def generate_examples(num_of_examples, verbose=False):
    """
    generate a function, inputs, outputs triple
    """

    prog = expression.Program.generate()
    inputs = []
    outputs = []

    # generate num_of_examples
    success_cnt = 0
    for _ in range(20):
        try:
            # we already have desired number of I/O examples
            if success_cnt == num_of_examples:
                # print('Generated program:', prog)
                return prog, inputs, outputs

            # we generate new inputs first, operate the program over the input
            in_str = generate_valid_input(prog.constraints)
            # print('Input:', in_str)
            out_state = execute_actions(prog.to_action(),
                                        action.RobState.new([in_str], [""]))
            out_str = out_state.committed[0]
            # print('Output:', out_str)
            # print('Program:', prog)
            # ensure the output str within len range
            if len(out_str) > MAX_STR_LEN:
                continue
            # ensure the output str is not empty, to be interesting
            if len(out_str) == 0:
                continue

            # store the successful I/O pairs
            inputs.append(in_str)
            outputs.append(out_str)
            success_cnt += 1
        except Exception as e:
            # in verbose mode, print out the exception message
            if verbose:
                print('Error', e, 'encountered, retrying...')

    return generate_examples(num_of_examples, verbose)
