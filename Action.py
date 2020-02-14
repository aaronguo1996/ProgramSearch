"""
String Manipulation Grammar

Program
p := concat(e1, e2, e3, ...)
Expression
e := f | n | n1(n2) | n(f) | constStr(c)
Substring
f := substr(k1, k2)
   | GetSpan(r1, i1, y1, r2, i2, y2)
Nesting
n := getToken(t, i) | toCase(s)
   | replace(d1, d2) | trim()
   | getUpto(r) | getFrom(r)
   | getFirst(t, i) | getAll(t)
Regex
r := t1 | ... | tn | d1 | ... | dm
Type
t := Number | Word | Alphanum
   | AllCaps | PropCase | Lower
   | Digit | Char
Case
s := Proper | AllCaps | Lower
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

    def str_masks_to_np_default():
        np_masks = np.zeros(shape = (max(INDEX) + 2, max(POSITION_K) + 1))

    def str_masks_to_np(self, str1, state):
        return Action.str_masks_to_np_default()


class ToCase(Action):

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

    def __call__(self, pstate):
        new_scratch = [self.execute(x) for x in pstate.scratch]
        check_change(pstate.scratch, new_scratch)
        return RobState(pstate.inputs,
                        new_scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])
