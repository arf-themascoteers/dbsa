from si_module import SIModule


class DI(SIModule):
    def __init__(self):
        super().__init__(2,2)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        return i-j

    def _names(self):
        return ["i","j"]