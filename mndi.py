from si_module import SIModule


class MNDI(SIModule):
    def __init__(self):
        super().__init__(4,3)

    def _forward(self, outs):
        alpha = outs[3]
        r_is = outs[0]
        r_js = outs[1]
        r_ks = outs[2]
        diff = r_js - alpha*r_ks
        up = r_is - diff
        down = r_is + diff
        mndis = up/down
        return mndis

    def _names(self):
        return ["i","j","k","alpha"]
