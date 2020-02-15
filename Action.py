"""
String Manipulation Grammar

- [ ] it might work better if we separate the regex finding and the index
finding into two steps
- [ ] is it better if we only learn one action at each step?

"""
import numpy as np
import string
import re
import random
import traceback
import pickle

import Util
import RobState

class Action:

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.name == other.name

    @staticmethod
    def str_mask_to_np_default():
        np_masks = np.zeros(shape = (MAX_STR_LEN,))
        return np_masks

    def str_mask_to_np(self, str, pstate):
        return Action.str_mask_to_np_default()

    def execute(self, x):
        pass

    def __call__(self, pstate):
        if self.name == 'Commit':
            """
            For commit action, it plays as the concat operation
            append the new committed string to all the previous outputs
            all the following operations start from input strings
            """
            new_scratch = pstate.inputs
            new_committed = [x[0] + x[1] for x in zip(pstate.committed,
                                                      pstate.scratch)]
            check_change(pstate.committed, new_committed)
            # used when rolling out
            for c, o in zip(new_committed, pstate.outputs):
                if output == "":
                    continue
                elif not o.startswith(c):
                    raise ConcatError

            return RobState(pstate.inputs,
                            new_scratch,
                            new_committed,
                            pstate.outputs,
                            pstate.past_actions + [self])
        else:
            """
            For other actions except commit,
            execute the action over all the intermediate states
            and replace the old scratch states with the new ones
            """
            new_scratch = [self.execute(x) for x in pstate.scratch]
            check_change(pstate.scratch, new_scratch)
            return RobState(pstate.inputs,
                            new_scratch,
                            pstate.committed,
                            pstate.outputs,
                            pstate.past_actions + [self])


class ToCase(Action):
    """
    Case := Proper | AllCaps | Lower

    Transformation on letter cases
    """

    def __init__(self, s):
        self.name = f"ToCase({s})"
        self.s = s

    def execute(self, x):
        if self.s == "Proper":
            return x.title()
        elif self.s == "AllCaps":
            return x.upper()
        elif self.s == "Lower":
            return x.lower()
        else:
            assert 0, "unrecognized case type"

    @staticmethod
    def generate_actions():
        return [ToCase('Proper'),
                ToCase('AllCaps'),
                ToCase 'Lower']

class Replace(Action):
    """
    Replace(d1, d2)

    Replace one delimiter with another one
    """

    def __init__(self, d1, d2):
        self.name = f"Replace({d1},{d2})"
        self.d1 = d1
        self.d2 = d2

    def execute(self, x):
        return x.replace(self.d1, self.d2)

    @staticmethod
    def generate_actions():
        return [Replace(d1, d2) for d1 in DELIMITERS
                for d2 in DELIMITERS
                if d2 != d1]

    def str_mask_to_np(self, str, pstate):
        mask = Action.str_mask_to_np_default()
        for i, c in enumerate(str):
            mask[i] = 1 if c == self.d1 else 0
        return mask

class Substr(Action):
    """
    Substr(k1, k2)

    Get a substring between index k1 and k2
    """

    def __init__(self, k1, k2):
        self.name = f"Substr({k1},{k2})"
        self.k1 = k1
        self.k2 = k2

    def execute(self, x)
        return x[self.k1:self.k2]

    @staticmethod
    def generate_actions():
        return [Substr(k1, k2) for k1 in POSITION_K for k2 in POSITION_K]

    def str_mask_to_np(self, str, pstate):
        mask = Action.str_mask_to_np_default()
        mask[self.k1:self.k2] = 1
        return mask

class GetToken(Action):
    """
    GetToken(t, i)

    ith match of regex t in the string from beginning (end if i < 0)
    """

    def __init__(self, t, i):
        self.name = f"GetToken({t},{i})"
        self.t = t
        self.i = i

    def getMatchIndex(self, x):
        allMatches = re.finditer(REGEX[self.t], x)
        match = list(allMatches)[self.i]
        return match

    def execute(self, x):
        match = self.getMatchIndex(x)
        return x[match.start():match.end()]

    @staticmethod
    def generate_actions():
        regTypes = REGEX.keys()
        return [GetToken(t, i) for t in regTypes for i in INDEX]

    def str_mask_to_np(self, str, pstate):
        match = self.getMatchIndex(str)
        mask = Action.str_mask_to_np_default()
        mask[match.start():match:end()] = 1
        return mask

class GetUpTo(Action):
    """
    GetUpTo(r)

    if the end of first match over r has index i, return str[0..r]
    """

    def __init__(self, r):
        self.name = f"GetUpTo({r})"
        self.r = r

    def getMatch(self, x):
        allMatches = re.finditer(ALL_REGEX[self.r], x)
        match = list(allMatches)[0]
        return match

    def execute(self, x):
        match = self.getMatch(x)
        return x[:match.end()]

    @staticmethod
    def generate_actions():
        regTypes = ALL_REGEX.keys()
        return [GetUpTo(r) for r in regTypes]

    def str_mask_to_np(self, str, pstate):
        match = self.getMatch(str)
        mask = Action.str_mask_to_np_default()
        mask[:match.end()] = 1
        return mask

class GetFrom(Action):
    """
    GetFrom(r)

    if the end of last match over r has index j, return str[j..]
    """

    def __init__(self, r):
        self.name = f"GetFrom({r})"
        self.r = r

    def getMatch(self, x):
        allMatches = re.finditer(ALL_REGEX[self.r], x)
        match = list(allMatches)[-1]
        return match

    def execute(self, x):
        match = self.getMatch(x)
        return x[match.end():]

    @staticmethod
    def generate_actions():
        regTypes = ALL_REGEX.keys()
        return [GetFrom(r) for r in regTypes]

    def str_mask_to_np(self, str, pstate):
        match = self.getMatch(str)
        mask = Action.str_mask_to_np_default()
        mask[match.end():] = 1
        return mask

class GetFirst(Action):
    """
    GetFirst(t, i)

    Concat(s1, s2, .. si) where sj is the jth match of t in str
    """

    def __init__(self, t, i):
        self.name = f"GetFirst({t},{i})"
        self.t = t
        self.i = i

    def getMatch(self, x):
        allMatches = re.finditer(REGEX[self.t], x)
        match = list(allMatches)[:self.i + 1]
        return match

    def execute(self, x):
        matches = self.getMatch(x)
        strs = [x[m.start():m.end()] for m in matches]
        return ''.join(strs)

    @staticmethod
    def generate_actions():
        regTypes = REGEX.keys()
        return [GetFirst(t, i) for t in regTypes for i in INDEX]

    def str_mask_to_np(self, str, pstate):
        matches = self.getMatch(str)
        mask = Action.str_mask_to_np_default()
        for m in matches:
            mask[m.start():m.end()] = 1
        return mask

class GetAll(Action):
    """
    GetAll(t)

    Concat(s1, s2, ..., sm) where si is the ith match of t in str
    """

    def __init__(self, t):
        self.name = f'GetAll({t})'
        self.t = t

    def getMatch(self, x):
        allMatches = re.finditer(REGEX[self.t], x)
        return list(allMatches)

    def execute(self, x):
        matches = self.getMatch(x)
        strs = [x[m.start():m.end()] for m in matches]
        return ''.join(strs)

    @staticmethod
    def generate_actions():
        regTypes = REGEX.keys()
        return [GetAll(t) for t in regTypes]

    def str_mask_to_np(self, str, pstate):
        matches = self.getMatch(str)
        mask = Action.str_mask_to_np_default()
        for m in matches:
            mask[m.start():m.end()] = 1
        return mask

class GetSpan(Action):
    """
    GetSpan(r1, i1, y1, r2, i2, y2)

    p1 = y1 of i1th match of r1
    p2 = y2 of i2th match of r2
    return str[p1...p2]
    """

    def __init__(self, r1, i1, y1, r2, i2, y2):
        self.name = f'GetSpan({r1},{i1},{y1},{r2},{i2},{y2})'
        self.r = (r1, r2)
        self.i = (i1, i2)
        self.y = (y1, y2)

    def getP(self, x):
        matches1 = re.finditer(ALL_REGEX[self.r[0]], x)
        i1match = list(matches1)[self.i[0]]
        p1 = i1match.start() if self.y[0] == 'Start' else i1match.end()
        matches2 = re.finditer(ALL_REGEX[self.r[1]], x)
        i2match = list(matches2)[self.i[1]]
        p2 = i2match.start() if self.y[1] == 'Start' else i2match.end()
        return p1, p2

    def execute(self, x):
        p1, p2 = self.getP(x)
        return x[p1:p2]

    @staticmethod
    def generate_actions():
        return [GetSpan(r1, i1, y1, r2, i2, y2)
                for r1 in ALL_REGEX
                for i1 in INDEX
                for y1 in BOUNDARY
                for r2 in ALL_REGEX
                for i2 in INDEX
                for y2 in BOUNDARY]

    # pstate usually is only used when we separate operations into several
    # steps, keep it there just in case we change the design in the future
    def str_mask_to_np(self, str, pstate):
        p1, p2 = self.getP(str)
        mask = Action.str_mask_to_np_default()
        mask[p1:p2] = 1
        return mask

class ConstStr(Action):
    """
    constant string
    """

    def __init__(self, c):
        self.name = f'ConstStr({c})'
        self.c = c

    def execute(self, x):
        return self.c

    @staticmethod
    def generate_actions():
        return [Const(c) for c in CHARACTERS]

class Commit(Action):
    """
    commit the current string for concatenation
    """

    def __init__(self):
        self.name = 'Commit'

    def execute(self, x):
        pass

ALL_ACTION_TYPES = [ToCase,
                    Replace,
                    Substr,
                    GetToken,
                    GetUpTo,
                    GetFrom,
                    GetFirst,
                    GetAll,
                    GetSpan,
                    Const,
                    Commit
                   ]
ALL_ACTIONS = [x for action_type in ALL_ACTION_TYPES
               for x in action_type.generate_actions()]
