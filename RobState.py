import numpy as np

from Util import *

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
        render_past_buttons = render_options['render_past_buttons']
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
