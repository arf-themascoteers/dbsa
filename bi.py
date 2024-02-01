from si_module import SIModule


class BI(SIModule):
    def __init__(self, ):
        super().__init__(1, 1)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        return i

    def _names(self):
        return ["i"]