"""
String Manipulation Grammar

Program
p := concat(e1, e2, e3, ...)
Expression
e := f | n | n1(n2) | n(f) | constStr(c)
Substring
f :=
   | GetSpan(r1, i1, y1, r2, i2, y2)
Nesting
n := getToken(t, i) | trim()
   | getUpto(r) | getFrom(r)
   | getFirst(t, i) | getAll(t)
Regex
r := t1 | ... | tn | d1 | ... | dm
Type
t := Number | Word | Alphanum
   | AllCaps | PropCase | Lower
   | Digit | Char
Position
k := -100 ~ 100
Index
i := -5 ~ 5
Character
c := A-Za-z0-9!?@
Delimiter
d := &!#$%^&*()
Boundary
y := Start | End

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
        if self.name == "Commit":
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
        return [Replace(d1, d2) for d1 in DELIMITERS for d2 in DELIMITERS]

    def str_mask_to_np(self, str, pstate):
        mask = Action.str_mask_to_np_default()
        for i, c in enumerate(str):
            mask[i] = if c == self.d1 then 1 else 0
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


