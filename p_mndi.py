import torch
from si_module import SIModule


class P_MNDI(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(4,3,6,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        alpha = params[3]

        mndis = []
        mndis.append(self.out(i,j,k,alpha))
        mndis.append(self.out(i,k,j,alpha))
        mndis.append(self.out(j,i,k,alpha))
        mndis.append(self.out(j,k,i,alpha))
        mndis.append(self.out(k,i,j,alpha))
        mndis.append(self.out(k,j,i,alpha))

        mndis = torch.cat(mndis, dim=1)
        return mndis

    def out(self, i, j, k, alpha):
        diff = j - alpha*k
        up = i - diff
        down = i + diff
        mndis = up/down
        return mndis.reshape(-1,1)

    def _names(self):
        return ["i","j","k","alpha"]
