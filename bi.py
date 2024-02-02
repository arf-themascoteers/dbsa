from si_module import SIModule


class BI(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(1, 1, initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        return i

    def _names(self):
        return ["i"]