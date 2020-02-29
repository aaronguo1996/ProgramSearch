from action import RobState
from util import *
import sys, os

class RobEnv:
    def __init__(self, inputs, outputs,
                 render_kind={'render_past_actions':False}):
        self.inputs = inputs
        self.outputs = outputs
        self.render_kind = render_kind
        self.verbose = False

    def reset(self):
        self.done = False
        self.pstate = RobState.new(self.inputs, self.outputs)
        init_ob = self.pstate.to_np(self.render_kind)
        self.last_step = init_ob, 0.0, self.done
        return init_ob

    def copy(self):
        ret = RobEnv(self.inputs, self.outputs, self.render_kind)
        ret.done = self.done
        ret.pstate = self.pstate.copy()
        ret.last_step = self.last_step
        return ret

    def step(self, action):
        try:
            # execute the action over the previous state
            self.pstate = action(self.pstate)
            next_ob = self.pstate.to_np(self.render_kind)
            # check action sequences
            if len(self.pstate.past_actions) >= 2:
                prev_action, curr_action = self.pstate.past_actions[-2:]
                prev_action.check_next_action(curr_action)
        except Exception as e:
            # if the execution fails, possible reasons:
            # commit a wrong string, no change operation or other index error
            if self.verbose:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_tb.tb_frame.f_code.co_filename)
                print(exc_tb.tb_lineno)
                print('Caught error for', action.name, e)

            self.done = True
            # what state should we put here?? why do we design an empty string
            # pairs for the crash state? pretend it didn't happen?
            self.last_step = RobState.to_crash_np(self.render_kind), -1.0, True
            return self.last_step

        # assign rewards
        reward = 1.0 if self.pstate.committed == self.pstate.outputs else 0.0
        done = reward != 0.0

        commits = filter(lambda x: x.name == 'Commit',
                         self.pstate.past_actions)
        num_of_commits = len(list(commits))
        if num_of_commits >= N_EXPRS:
            done = True

        self.done = done
        self.last_step = next_ob, reward, done

        return self.last_step

