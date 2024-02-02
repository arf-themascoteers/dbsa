from si_module import SIModule


class SNDI(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(3,2, initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        alpha = params[2]
        scaled_j = alpha*j
        return (i - scaled_j)/(i + scaled_j)

    def _names(self):
        return ["i","j","alpha"]
