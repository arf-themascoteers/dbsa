from si_module import SIModule


class SNDI(SIModule):
    def __init__(self):
        super().__init__(3,2)

    def _forward(self, outs):
        alpha = outs[2]
        scaled_j = alpha*outs[1]
        return (outs[0] - scaled_j)/(outs[0] + scaled_j)

    def _names(self):
        return ["i","j","alpha"]
