import string
import random
import re

import action
import pregex as pre
from util import *

class Program:
    """
    concatenation of expressions
    """

    def __init__(self, exprs):
        self.exprs = exprs
        constraints = [e.constraints for e in exprs]
        self.constraints = merge_constraints(constraints)

    @staticmethod
    def generate():
        num_of_concat = random.randint(1, N_EXPRS)
        return Program([Expression.generate() for i in range(num_of_concat)])

    def eval_str(self, str):
        return ''.join([e.eval_str(str) for e in self.exprs])

    def to_action(self):
        actions = []

        for e in self.exprs:
            actions += e.to_action()
            actions.append(action.Commit())

        return actions

    def __str__(self):
        str_exprs = [str(e) for e in self.exprs]
        return ' | '.join(str_exprs)

class Expression:
    """
    substring | nesting | nesting of nesting | nesting of substring | constant
    """

    def __init__(self, e):
        self.e = e
        self.constraints = e.constraints

    @staticmethod
    def generate():
        choices = [
            lambda: Substring.generate(),
            lambda: Nesting.generate(),
            lambda: Composition(Nesting.generate(), Nesting.generate()),
            lambda: Composition(Nesting.generate(), Substring.generate()),
            lambda: Constant.generate()
        ]
        return Expression(random.choice(choices)())

    def eval_str(self, str):
        return self.e.eval_str(str)

    def to_action(self):
        return self.e.to_action()

    def __str__(self):
        return str(self.e)

class Composition:
    """
    composition of substrings or nestings
    """

    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2
        self.constraints = merge_constraints([e1.constraints, e2.constraints])

    def eval_str(self, str):
        return self.e1.eval_str(self.e2.eval_str(str))

    def to_action(self):
        return self.e2.to_action() + self.e1.to_action()

    def __str__(self):
        return str(f'{str(self.e1)}({str(self.e2)})')

class Constant:
    """
    a constant character
    """

    def __init__(self, c):
        self.c = c
        self.constraints = ({}, 0)

    @staticmethod
    def generate():
        l = random.choice(list(range(1, MAX_CONST_LEN)))
        c = pre.create("."*l).sample()
        return Constant(c)

    def eval_str(self, str):
        return self.c

    def to_action(self):
        return [action.ConstStr(self.c)]

    def __str__(self):
        return f'Const({self.c})'

class Substring:
    """
    get a substring either by indices or by regex matching
    """

    @staticmethod
    def generate():
        choices = [
            SubstrIndex,
            SubstrSpan
        ]
        return random.choice(choices).generate()

class SubstrIndex:
    """
    get a substring by start and end indices
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.constraints = ({}, max(abs(k1), abs(k2)))

    @staticmethod
    def generate():
        k1 = random.choice(POSITION_K)
        k2 = random.choice(POSITION_K)
        if k1 > k2:
            k1, k2 = k2, k1
        return SubstrIndex(k1, k2)

    def eval_str(self, str):
        return str[self.k1:self.k2]

    def to_action(self):
        return [action.Substr(self.k1, self.k2)]

    def __str__(self):
        return f'Substr({self.k1},{self.k2})'

class SubstrSpan:
    """
    get a substring by span of regular expressions
    """

    def __init__(self, r1, i1, y1, r2, i2, y2):
        self.r = (r1, r2)
        self.i = (i1, i2)
        self.y = (y1, y2)
        reg_constraints = {
            r1: step_abs(i1),
            r2: step_abs(i2)
        }
        self.constraints = (reg_constraints, 0)

    @staticmethod
    def generate():
        r1 = RegexGen.generate()
        r2 = RegexGen.generate()
        i1 = random.choice(INDEX)
        i2 = random.choice(INDEX)
        y1 = random.choice(BOUNDARY)
        y2 = random.choice(BOUNDARY)
        return SubstrSpan(r1, i1, y1, r2, i2, y2)

    def eval_str(self, str):
        matches1 = re.finditer(REGEX[self.r[0]], str)
        match1 = list(matches1)[self.i[0]]
        p1 = match1.start() if self.y[0] == 'Start' else match1.end()
        matches2 = re.finditer(REGEX[self.r[1]], str)
        match2 = list(matches2)[self.i[1]]
        p2 = match2.start() if self.y[1] == 'Start' else match2.end()
        return str[p1:p2]

    def to_action(self):
        return [action.GetSpan(self.r[0], self.i[0], self.y[0],
                               self.r[1], self.i[1], self.y[1])]

    def __str__(self):
        return f'Span({self.r[0]},{self.i[0]},{self.y[0]},{self.r[1]},{self.i[1]},{self.y[1]})'

class Nesting:
    """
    nest of strings
    """

    @staticmethod
    def generate():
        choices = [
            GetToken,
            ToCase,
            Replace,
            GetUpTo,
            GetFrom,
            GetFirst,
            GetAll
        ]
        return random.choice(choices).generate()

class GetToken:
    """
    get token from beginning to the end of ith match of t
    """

    def __init__(self, t, i):
        self.t = t
        self.i = i
        self.constraints = ({t: step_abs(i)}, 0)

    @staticmethod
    def generate():
        t = RegexGen.generate_type()
        i = random.choice(INDEX)
        return GetToken(t, i)

    def eval_str(self, str):
        allMatches = re.finditer(REGEX[self.t], str)
        match = list(allMatches)[self.i]
        return str[match.start():match.end()]

    def to_action(self):
        return [action.GetToken(self.t, self.i)]

    def __str__(self):
        return f'GetToken({self.t},{self.i})'

class ToCase:
    """
    transform the case of a string
    """

    def __init__(self, s):
        self.s = s
        self.constraints = ({'Alphanum': 1}, 1)

    @staticmethod
    def generate():
        s = random.choice(CASES)
        return ToCase(s)

    def eval_str(self, str):
        if self.s == 'Proper':
            return str.title()
        elif self.s == 'AllCaps':
            return str.upper()
        elif self.s == 'Lower':
            return str.lower()
        else:
            raise ValueException

    def to_action(self):
        return [action.ToCase(self.s)]

    def __str__(self):
        return f'ToCase({self.s})'

class Replace:
    """
    replace a delimter with another
    """

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        self.constraints = ({d1: 1}, 1)

    @staticmethod
    def generate():
        d1 = random.choice(DELIMITERS)
        d2 = random.choice([d for d in DELIMITERS if d != d1])
        return Replace(d1, d2)

    def eval_str(self, str):
        return str.replace(self.d1, self.d2)

    def to_action(self):
        return [action.Replace(self.d1, self.d2)]

    def __str__(self):
        return f'Replace({self.d1},{self.d2})'

class GetUpTo:
    """
    get a substring from the beginning to the matched regex
    """

    def __init__(self, r):
        self.r = r
        self.constraints = ({r: 1}, 1)

    @staticmethod
    def generate():
        r = RegexGen.generate()
        return GetUpTo(r)

    def eval_str(self, str):
        allMatches = re.finditer(ALL_REGEX[self.r], str)
        match = list(allMatches)[0]
        return str[:match.end()]

    def to_action(self):
        return [action.GetUpTo(self.r)]

    def __str__(self):
        return f'GetUpto({self.r})'

class GetFrom:
    """
    get a substring from the end of last match of r to the end of string
    """

    def __init__(self, r):
        self.r = r
        self.constraints = ({r: 1}, 1)

    @staticmethod
    def generate():
        r = RegexGen.generate()
        return GetFrom(r)

    def eval_str(self, str):
        allMatches = re.finditer(ALL_REGEX[self.r], str)
        match = list(allMatches)[-1]
        return str[match.end():]

    def to_action(self):
        return [action.GetFrom(self.r)]

    def __str__(self):
        return f'GetFrom({self.r})'

class GetFirst:
    """
    concatenate the first i matches of the given regex t
    """

    def __init__(self, t, i):
        self.t = t
        self.i = i
        self.constraints = ({t: step_abs(i)}, 1)

    @staticmethod
    def generate():
        t = RegexGen.generate_type()
        i = random.choice(INDEX)
        return GetFirst(t, i)

    def eval_str(self, str):
        allMatches = re.finditer(REGEX[self.t], str)
        matches = list(allMatches)[:self.i+1]
        return ''.join([str[m.start():m.end()] for m in matches])

    def to_action(self):
        return [action.GetFirst(self.t, self.i)]

    def __str__(self):
        return f'GetFirst({self.t},{self.i})'

class GetAll:
    """
    concatenate all the matches of the given regex t
    """

    def __init__(self, t):
        self.t = t
        self.constraints = ({t: 1}, 1)

    @staticmethod
    def generate():
        t = RegexGen.generate_type()
        return GetAll(t)

    def eval_str(self, str):
        allMatches = re.finditer(REGEX[self.t], str)
        return ''.join([str[m.start():m.end()] for m in allMatches])

    def to_action(self):
        return [action.GetAll(self.t)]

    def __str__(self):
        return f'GetAll({self.t})'

class RegexGen:
    @staticmethod
    def generate_type():
        type_choice = random.choice(list(REGEX.keys()))
        return type_choice

    @staticmethod
    def generate_delim():
        delim_choice = random.choice(list(DELIM_REGEX.keys()))
        return delim_choice

    @staticmethod
    def generate():
        if np.random.random() < 0.5:
            return RegexGen.generate_type()
        else:
            return RegexGen.generate_delim()
