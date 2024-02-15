import torch
from si_module import SIModule
from gndi3 import GNDI3


class GNDI4(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(7,4,24,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        l = splines.evaluate(params[3])
        alpha1 = params[4]
        alpha2 = params[5]
        alpha3 = params[6]

        gndis = []
        
        gndis.append(self.out(i,j,k,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(i,j,l,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(i,k,j,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(i,k,l,j,alpha1,alpha2,alpha3))
        gndis.append(self.out(i,l,j,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(i,l,k,j,alpha1,alpha2,alpha3))

        gndis.append(self.out(j,i,k,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(j,i,l,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(j,k,i,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(j,k,l,i,alpha1,alpha2,alpha3))
        gndis.append(self.out(j,l,i,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(j,l,k,i,alpha1,alpha2,alpha3))

        gndis.append(self.out(k,j,i,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(k,j,l,i,alpha1,alpha2,alpha3))
        gndis.append(self.out(k,i,j,l,alpha1,alpha2,alpha3))
        gndis.append(self.out(k,i,l,j,alpha1,alpha2,alpha3))
        gndis.append(self.out(k,l,j,i,alpha1,alpha2,alpha3))
        gndis.append(self.out(k,l,i,j,alpha1,alpha2,alpha3))

        gndis.append(self.out(l,j,k,i,alpha1,alpha2,alpha3))
        gndis.append(self.out(l,j,i,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(l,k,j,i,alpha1,alpha2,alpha3))
        gndis.append(self.out(l,k,i,j,alpha1,alpha2,alpha3))
        gndis.append(self.out(l,i,j,k,alpha1,alpha2,alpha3))
        gndis.append(self.out(l,i,k,j,alpha1,alpha2,alpha3))

        gndis = torch.cat(gndis, dim=1)
        return gndis

    @staticmethod
    def out(i, j, k, l, alpha1, alpha2, alpha3):
        result = GNDI4.num(i, j, k, l, alpha1, alpha2, alpha3) / GNDI4.den(i, j, k, l, alpha1, alpha2, alpha3)
        return result.reshape(-1,1)

    @staticmethod
    def num(i,j,k,l,alpha1,alpha2,alpha3):
        prev_num = GNDI3.num(i,j,k,alpha1,alpha2)
        return l - alpha3 * prev_num

    @staticmethod
    def den(i,j,k,l,alpha1,alpha2,alpha3):
        prev_den = GNDI3.den(i,j,k,alpha1,alpha2)
        return l + alpha3 * prev_den

    def _names(self):
        return ["i","j","k","l","alpha1","alpha2","alpha3"]
