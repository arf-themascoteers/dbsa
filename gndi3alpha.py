import torch
from si_module import SIModule
from gndi2 import GNDI2


class GNDI3Alpha(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(15,3,6,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        alpha1_1 = params[3]
        alpha1_2 = params[4]
        alpha1_3 = params[5]
        alpha1_4 = params[6]
        alpha1_5 = params[7]
        alpha1_6 = params[8]
        alpha2_1 = params[9]
        alpha2_2 = params[10]
        alpha2_3 = params[11]
        alpha2_4 = params[12]
        alpha2_5 = params[13]
        alpha2_6 = params[14]

        gndis = []
        
        gndis.append(self.out(i,j,k,alpha1_1, alpha2_1))
        gndis.append(self.out(i,k,j,alpha1_2, alpha2_2))
        gndis.append(self.out(j,i,k,alpha1_3, alpha2_3))
        gndis.append(self.out(j,k,i,alpha1_4, alpha2_4))
        gndis.append(self.out(k,i,j,alpha1_5, alpha2_5))
        gndis.append(self.out(k,j,i,alpha1_6, alpha2_6))

        gndis = torch.cat(gndis, dim=1)
        return gndis

    @staticmethod
    def out(i, j, k, alpha1, alpha2):
        result = GNDI3Alpha.num(i, j, k, alpha1, alpha2) / GNDI3Alpha.den(i, j, k, alpha1, alpha2)
        return result.reshape(-1,1)

    @staticmethod
    def num(i,j,k,alpha1,alpha2):
        prev_num = GNDI2.num(i,j,alpha1)
        return k - alpha2 * prev_num

    @staticmethod
    def den(i,j,k,alpha1,alpha2):
        prev_den = GNDI2.den(i,j,alpha1)
        return k + alpha2 * prev_den

    def _names(self):
        return ["i","j","k",
                "alpha1_1","alpha1_2","alpha1_3","alpha1_4","alpha1_5","alpha1_6",
                "alpha2_1","alpha2_2","alpha2_3","alpha2_4","alpha2_5", "alpha2_6"
                ]
