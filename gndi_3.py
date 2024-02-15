import torch
from si_module import SIModule
from gndi_2 import GNDI_2


class GNDI_3(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(5,3,6,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        alpha1 = params[3]
        alpha2 = params[3]

        gndis = []
        
        gndis.append(self.out(i,j,k,alpha1, alpha2))
        gndis.append(self.out(i,k,j,alpha1, alpha2))
        gndis.append(self.out(j,i,k,alpha1, alpha2))
        gndis.append(self.out(j,k,i,alpha1, alpha2))
        gndis.append(self.out(k,i,j,alpha1, alpha2))
        gndis.append(self.out(k,j,i,alpha1, alpha2))

        gndis = torch.cat(gndis, dim=1)
        return gndis

    @staticmethod
    def out(i, j, k, alpha1, alpha2):
        result = GNDI_3.num(i,j,k,alpha1,alpha2) / GNDI_3.den(i,j,k,alpha1,alpha2)
        return result.reshape(-1,1)

    @staticmethod
    def num(i,j,k,alpha1,alpha2):
        prev_num = GNDI_2.num(i,j,alpha1)
        return k - alpha2 * prev_num

    @staticmethod
    def den(i,j,k,alpha1,alpha2):
        prev_den = GNDI_2.den(i,j,alpha1)
        return k + alpha2 * prev_den

    def _names(self):
        return ["i","j","k","alpha1","alpha2"]
