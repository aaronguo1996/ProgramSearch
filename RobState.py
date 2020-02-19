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
