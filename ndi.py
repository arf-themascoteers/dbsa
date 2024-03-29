from si_module import SIModule


class NDI(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(2,2,1, initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        return ((i-j)/(i+j)).reshape(-1,1)

    def _names(self):
        return ["i","j"]