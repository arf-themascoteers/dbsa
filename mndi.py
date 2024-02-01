from si_module import SIModule


class MNDI(SIModule):
    def __init__(self):
        super().__init__(4,3)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        alpha = params[3]

        diff = j - alpha*k
        up = i - diff
        down = i + diff
        mndis = up/down
        return mndis

    def _names(self):
        return ["i","j","k","alpha"]
