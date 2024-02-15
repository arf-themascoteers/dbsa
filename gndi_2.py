import torch
from si_module import SIModule


class GNDI_2(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(3,2,2,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        alpha1 = params[2]

        gndis = []
        gndis.append(GNDI_2.out(i, j, alpha1))
        gndis.append(GNDI_2.out(j, i, alpha1))
        gndis = torch.cat(gndis, dim=1)
        return gndis

    @staticmethod
    def out(i, j, alpha1):
        result = GNDI_2.num(i, j, alpha1) / GNDI_2.den(i, j, alpha1)
        return result.reshape(-1,1)

    @staticmethod
    def num(i, j, alpha1):
        return i-alpha1*j

    @staticmethod
    def den(i, j, alpha1):
        return i+alpha1*j

    def _names(self):
        return ["i","j","k","alpha1"]
