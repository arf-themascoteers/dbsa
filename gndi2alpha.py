import torch
from si_module import SIModule


class GNDI2Alpha(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(4,2,2,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        alpha1_1 = params[2]
        alpha1_2 = params[3]

        gndis = []
        gndis.append(GNDI2Alpha.out(i, j, alpha1_1))
        gndis.append(GNDI2Alpha.out(j, i, alpha1_2))
        gndis = torch.cat(gndis, dim=1)
        return gndis

    @staticmethod
    def out(i, j, alpha1):
        result = GNDI2Alpha.num(i, j, alpha1) / GNDI2Alpha.den(i, j, alpha1)
        return result.reshape(-1,1)

    @staticmethod
    def num(i, j, alpha1):
        return i-alpha1*j

    @staticmethod
    def den(i, j, alpha1):
        return i+alpha1*j

    def _names(self):
        return ["i","j","alpha1_1","alpha1_2"]
