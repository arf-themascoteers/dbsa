from si_module import SIModule


class BI(SIModule):
    def __init__(self, initial_value):
        super().__init__(1, initial_value)

    def _forward(self, outs):
        return outs[0]

    def _names(self):
        return ["band"]