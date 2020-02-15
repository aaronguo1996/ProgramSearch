import random
import string
import re

import Expression
import Util

class ProgramSampler:
    """
    random sampling of programs
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_program():
        """
        generate a random string manipulation program
        """
        return Expression.Program.generate()

    @staticmethod
    def generate_valid_input(constraints, max_string_size=MAX_STR_LEN):
        """
        generate a random but valid input string
        """
        pass

def generate_examples(num_of_examples, verbose=False):
    """
    generate a function, inputs, outputs triple
    """

    prog = P.generate()
    inputs = []
    outputs = []

    
