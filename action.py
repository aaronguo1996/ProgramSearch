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

from util import *
from error import *

class Action:

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.name == other.name

    @staticmethod
    def str_mask_to_np_default():
        np_masks = np.zeros(shape = (MAX_MASK_LEN, MAX_STR_LEN))
        return np_masks

    def str_mask_to_np(self, str, pstate):
        return Action.str_mask_to_np_default()

    def execute(self, pstate, x):
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
                if o == "":
                    continue
                elif not o.startswith(c):
                    print(o, c)
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
            new_scratch = [self.execute(pstate, x) for x in pstate.scratch]
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

    def execute(self, pstate, x):
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
                ToCase('Lower')]

class Replace1(Action):
    """
    Replace(d1, d2)

    Replace one delimiter with another one
    """

    def __init__(self, d1):
        self.name = f"Replace1({d1})"
        self.d1 = d1

    def check_next_action(self, next_action):
        if "Replace2" not in next_action.name:
            raise ActionSeqError

    @staticmethod
    def generate_actions():
        return [Replace1(d1) for d1 in DELIMITERS]

    # -2 is used for replace
    def str_mask_to_np(self, str, pstate):
        mask = Action.str_mask_to_np_default()
        idxs = [pos for pos, char in enumerate(str1) if char == self.d1]
        mask[-2][idxs] = 1
        return mask

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class Replace2(Action):
    def __init__(self, d2):
        self.name = f'Replace2({d2})'
        self.d2 = d2

    @staticmethod
    def generate_actions():
        return [Replace2(d2) for d2 in DELIMITERS]

    def execute(self, pstate, x):
        if "Replace1" not in pstate.past_actions[-1].name:
            raise ActionSeqError

        d1 = pstate.past_actions[-1].d1
        return x.replace(d1, self.d2)

class Substr1(Action):
    """
    Substr(k1, k2)

    Get a substring between index k1 and k2
    """

    def __init__(self, k1):
        self.name = f"Substr1({k1})"
        self.k1 = k1

    @staticmethod
    def generate_actions():
        return [Substr1(k1) for k1 in POSITION_K]

    def check_next_action(self, next_action):
        if "Substr2" not in next_action.name:
            raise ActionSeqError

    # -1 mask is used for substr
    def str_mask_to_np(self, str, pstate):
        mask = Action.str_mask_to_np_default()
        mask[-1][self.k1:] = 1
        return mask

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class Substr2(Action):
    def __init__(self, k2):
        self.name = f'Substr2({k2})'
        self.k2 = k2

    @staticmethod
    def generate_actions():
        return [Substr2(k2) for k2 in POSITION_K]

    def execute(self, pstate, x):
        if "Substr1" not in pstate.past_actions[-1].name:
            raise ActionSeqError

        k1 = pstate.past_actions[-1].k1
        return x[k1:self.k2]

class GetToken1(Action):
    """
    GetToken(t, i)

    ith match of regex t in the string from beginning (end if i < 0)
    """

    def __init__(self, t):
        self.name = f"GetToken1({t})"
        self.t = t

    @staticmethod
    def generate_actions():
        return [GetToken1(t) for t in REGEX.keys()]

    def check_next_action(self, next_action):
        if "GetToken2" not in next_action.name:
            raise ActionSeqError

    def str_masks_to_np(self, str, pstate):
        masks = Action.str_mask_to_np_default()
        # enumerate over all the regex masks
        p = list(re.finditer(REGEX[self.t], str))
        # first 5 masks are used for regex
        for i, m in enumerate(p[:max(INDEX)]):
            masks[i][m.start():m.end()] = 1
        return masks

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetToken2(Action):
    def __init__(self, i):
        self.name = f'GetToken2({i})'
        self.i = i

    def execute(self, pstate, x):
        if "GetToken1" not in pstate.past_actions[-1].name:
            raise ActionSeqError

        t = pstate.past_actions[-1].t
        allMatches = re.finditer(REGEX[t], x)
        match = list(allMatches)[self.i]
        return x[match.start():match.end()]

    @staticmethod
    def generate_actions():
        return [GetToken2(i) for i in INDEX]

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

    def execute(self, pstate, x):
        match = self.getMatch(x)
        return x[:match.end()]

    @staticmethod
    def generate_actions():
        regTypes = ALL_REGEX.keys()
        return [GetUpTo(r) for r in regTypes]

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

    def execute(self, pstate, x):
        match = self.getMatch(x)
        return x[match.end():]

    @staticmethod
    def generate_actions():
        regTypes = ALL_REGEX.keys()
        return [GetFrom(r) for r in regTypes]

class GetFirst1(Action):
    """
    GetFirst(t, i)

    Concat(s1, s2, .. si) where sj is the jth match of t in str
    """

    def __init__(self, t):
        self.name = f"GetFirst1({t})"
        self.t = t

    @staticmethod
    def generate_actions():
        return [GetFirst1(t) for t in REGEX.keys()]

    def str_mask_to_np(self, str, pstate):
        matches = list(re.finditer(self.t, str))
        masks = Action.str_mask_to_np_default()
        for i, m in enumerate(matches[:max(INDEX)]):
            masks[i][m.start():m.end()] = 1
        return masks

    def check_next_action(self, next_action):
        if "GetFirst2" not in next_action.name:
            raise ActionSeqError

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetFirst2(Action):
    def __init__(self, i):
        self.name = f'GetFirst2({i})'
        self.i = i

    @staticmethod
    def generate_actions():
        return [GetFirst2(i) for i in INDEX]

    def execute(self, pstate, x):
        if "GetFirst1" not in pstate.past_actions[-1].name:
            raise ActionSeqError

        t = pstate.past_actions[-1].t
        allMatches = re.finditer(REGEX[t], x)
        firstMatches = list(allMatches)[:self.i]
        strs = [x[m.start():m.end()] for m in firstMatches]
        return ''.join(strs)

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

    def execute(self, pstate, x):
        matches = self.getMatch(x)
        strs = [x[m.start():m.end()] for m in matches]
        return ''.join(strs)

    @staticmethod
    def generate_actions():
        regTypes = REGEX.keys()
        return [GetAll(t) for t in regTypes]

class GetSpan1(Action):
    """
    GetSpan(r1, i1, y1, r2, i2, y2)

    p1 = y1 of i1th match of r1
    p2 = y2 of i2th match of r2
    return str[p1...p2]
    """

    def __init__(self, r1):
        self.name = f'GetSpan1({r1})'
        self.r1 = r1

    @staticmethod
    def generate_actions():
        return [GetSpan1(r) for r in ALL_REGEX.keys()]

    def check_next_action(self, next_action):
        if 'GetSpan2' not in next_action.name:
            raise ActionSeqError

    def str_mask_to_np(self, str, pstate):
        # mask for 1st reg
        masks = Action.str_mask_to_np_default()
        matches = list(re.finditer(ALL_REGEX[self.r1], str))
        for i, m in enumerate(matches[:max(INDEX)]):
            masks[i][m.start():] = 1
        return masks

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetSpan2(Action):
    def __init__(self, i1):
        self.name = f'GetSpan2({i1})'
        self.i1 = i1

    def check_next_action(self, next_action):
        if "GetSpan3" not in next_action.name:
            raise ActionSeqError

    @staticmethod
    def generate_actions():
        return [GetSpan2(i) for i in INDEX]

    def str_mask_to_np(self, str, pstate):
        # mask for 1st reg
        r1 = pstate.past_actions[-1].r1
        masks = Action.str_mask_to_np_default()
        matches = list(re.finditer(ALL_REGEX[r1], str))
        m = matches[self.i1]
        masks[-1][m.start():] = 1
        return masks

    def __call__(self, pstate):
        if 'GetSpan1' not in pstate.past_actions[-1].name:
            raise ActionSeqError

        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetSpan3(Action):
    def __init__(self, y1):
        self.name = f'GetSpan3({y1})'
        self.y1 = y1

    @staticmethod
    def generate_actions():
        return [GetSpan3(y) for y in BOUNDARY]

    def check_next_action(self, next_action):
        if 'GetSpan4' not in next_action.name:
            raise ActionSeqError

    def str_mask_to_np(self, str, pstate):
        r1 = pstate.past_actions[-2].r1
        i1 = pstate.past_actions[-1].i1
        matches = list(re.finditer(ALL_REGEX[r1], str))
        m = matches[i1]
        masks = Action.str_mask_to_np_default()
        masks[-1][m.start():] = 1
        if self.y1 == 'End':
            masks[-1][m.start():m.end()] = 0
        return masks

    def __call__(self, pstate):
        if 'GetSpan2' not in pstate.past_actions[-1].name:
            raise ActionSeqError

        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetSpan4(Action):
    def __init__(self, r2):
        self.name = f'GetSpan4({r2})'
        self.r2 = r2

    @staticmethod
    def generate_actions():
        return [GetSpan4(r) for r in ALL_REGEX.keys()]

    def check_next_action(self, next_action):
        if 'GetSpan5' not in next_action.name:
            raise ActionSeqError

    def str_mask_to_np(self, str, pstate):
        r1 = pstate.past_actions[-3].r1
        i1 = pstate.past_actions[-2].i1
        y1 = pstate.past_actions[-1].y1
        matches = list(re.finditer(ALL_REGEX[r1], str))
        m = matches[i1]
        masks = Action.str_mask_to_np_default()
        masks[-1][m.start():] = 1
        if self.y1 == 'End':
            masks[-1][m.start():m.end()] = 0

        matches = list(re.finditer(ALL_REGEX[self.r2], str))
        for i, m in enumerate(matches[:max(INDEX)]):
            masks[i][:m.end()] = 1
        return masks

    def __call__(self, pstate):
        if 'GetSpan3' not in pstate.past_actions[-1].name:
            raise ActionSeqError

        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetSpan5(Action):
    def __init__(self, i2):
        self.name = f'GetSpan5({i2})'
        self.i2 = i2

    @staticmethod
    def generate_actions():
        return [GetSpan5(i) for i in INDEX]

    def check_next_action(self, next_action):
        if 'GetSpan6' not in next_action.name:
            raise ActionSeqError

    def str_mask_to_np(self, str, pstate):
        r1 = pstate.past_actions[-4].r1
        i1 = pstate.past_actions[-3].i1
        y1 = pstate.past_actions[-2].y1
        r2 = pstate.past_actions[-1].r2
        matches = list(re.finditer(ALL_REGEX[r1], str))
        m = matches[i1]
        masks = Action.str_mask_to_np_default()
        masks[-1][m.start():] = 1
        if self.y1 == 'End':
            masks[-1][m.start():m.end()] = 0

        matches = list(re.finditer(ALL_REGEX[r2], str))
        m = matches[self.i2]
        tmp_mask = np.zeros(shape = (MAX_STR_LEN,))
        tmp_mask[:m.end()] = 1
        masks[-1] = masks[-1] * tmp_mask

        return masks

    def __call__(self, pstate):
        if 'GetSpan4' not in pstate.past_actions[-1].name:
            raise ActionSeqError

        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_actions + [self])

class GetSpan6(Action):
    def __init__(self, y2):
        self.name = f'GetSpan6({y2})'
        self.y2 = y2

    @staticmethod
    def generate_actions():
        return [GetSpan6(y) for y in BOUNDARY]

    def execute(self, pstate, x):
        if 'GetSpan5' not in pstate.past_actions[-1].name:
            raise ActionSeqError

        r1 = pstate.past_actions[-5].r1
        i1 = pstate.past_actions[-4].i1
        y1 = pstate.past_actions[-3].y1
        r2 = pstate.past_actions[-2].r2
        i2 = pstate.past_actions[-1].i2
        matches = list(re.finditer(ALL_REGEX[r1], x))
        m = matches[i1]
        p1 = m.end() if y1 == 'End' else m.start()

        matches = list(re.finditer(ALL_REGEX[r2], x))
        m = matches[i2]
        p2 = m.end() if self.y2 == 'End' else m.start()

        return x[p1:p2]

class ConstStr(Action):
    """
    constant string
    """

    def __init__(self, c):
        self.name = f'ConstStr({c})'
        self.c = c

    def execute(self, pstate, x):
        return self.c

    @staticmethod
    def generate_actions():
        return [ConstStr(c) for c in CHARACTERS]

class Commit(Action):
    """
    commit the current string for concatenation
    """

    def __init__(self):
        self.name = 'Commit'

    def execute(self, pstate, x):
        pass

    @staticmethod
    def generate_actions():
        return [Commit()]

ALL_ACTION_TYPES = [ToCase,
                    Replace1,
                    Replace2,
                    Substr1,
                    Substr2,
                    GetToken1,
                    GetToken2,
                    GetUpTo,
                    GetFrom,
                    GetFirst1,
                    GetFirst2,
                    GetAll,
                    GetSpan1,
                    GetSpan2,
                    GetSpan3,
                    GetSpan4,
                    GetSpan5,
                    GetSpan6,
                    ConstStr,
                    Commit
                   ]
ALL_ACTIONS = [x for action_type in ALL_ACTION_TYPES
               for x in action_type.generate_actions()]

class RobState:

    def __init__(self, inputs, scratch, committed, outputs, past_actions):
        self.inputs = [x for x in inputs]
        self.scratch = [x for x in scratch]
        self.committed = [x for x in committed]
        self.outputs = [x for x in outputs]
        self.past_actions = [x for x in past_actions]

    def __repr__(self):
        return str((self.inputs,
                    self.scratch,
                    self.committed,
                    self.outputs,
                    self.past_actions))

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def new(inputs, outputs):
        scratch = [x for x in inputs]
        committed = ["" for _ in inputs]
        return RobState(inputs, scratch, committed, outputs, [])

    def copy(self):
        return RobState(self.inputs,
                        self.scratch,
                        self.committed,
                        self.outputs,
                        self.past_actions)

    def get_last_action_type(self):
        if self.past_actions == []:
            return 0
        else:
            return ALL_ACTION_TYPES.index(self.past_actions[-1].__class__) + 1

    def to_np(self, render_options):
        if self.past_actions == []:
            masks = [Action.str_mask_to_np_default()
                     for _ in range(len(self.inputs))]
        else:
            masks = [self.past_actions[-1].str_mask_to_np(s, self)
                     for s in self.scratch]

        rendered_inputs = str_to_np(self.inputs)
        rendered_scratch = str_to_np(self.scratch)
        rendered_committed = str_to_np(self.committed)
        rendered_outputs = str_to_np(self.outputs)
        rendered_masks = np.array(masks)
        rendered_last_action_type = self.get_last_action_type()

        return (rendered_inputs,
                rendered_scratch,
                rendered_committed,
                rendered_outputs,
                rendered_masks,
                rendered_last_action_type
               )

    @staticmethod
    def to_crash_np(render_kind):
        return RobState.new(["" for _ in range(N_IO)],
                            ["" for _ in range(N_IO)]).to_np(render_kind)
