import torch
from si_module import SIModule
from gndi3 import GNDI3


class GNDI4(SIModule):
    def __init__(self, initial_value=None):
        super().__init__(76,4,24,initial_value)

    def _forward(self, splines, params):
        i = splines.evaluate(params[0])
        j = splines.evaluate(params[1])
        k = splines.evaluate(params[2])
        l = splines.evaluate(params[3])

        alpha1_1 = params[4]
        alpha1_2 = params[5]
        alpha1_3 = params[6]
        alpha1_4 = params[7]
        alpha1_5 = params[8]
        alpha1_6 = params[9]
        alpha1_7 = params[10]
        alpha1_8 = params[11]
        alpha1_9 = params[12]
        alpha1_10 = params[13]
        alpha1_11 = params[14]
        alpha1_12 = params[15]
        alpha1_13 = params[16]
        alpha1_14 = params[17]
        alpha1_15 = params[18]
        alpha1_16 = params[19]
        alpha1_17 = params[20]
        alpha1_18 = params[21]
        alpha1_19 = params[22]
        alpha1_20 = params[23]
        alpha1_21 = params[24]
        alpha1_22 = params[25]
        alpha1_23 = params[26]
        alpha1_24 = params[27]
        alpha2_1 = params[28]
        alpha2_2 = params[29]
        alpha2_3 = params[30]
        alpha2_4 = params[31]
        alpha2_5 = params[32]
        alpha2_6 = params[33]
        alpha2_7 = params[34]
        alpha2_8 = params[35]
        alpha2_9 = params[36]
        alpha2_10 = params[37]
        alpha2_11 = params[38]
        alpha2_12 = params[39]
        alpha2_13 = params[40]
        alpha2_14 = params[41]
        alpha2_15 = params[42]
        alpha2_16 = params[43]
        alpha2_17 = params[44]
        alpha2_18 = params[45]
        alpha2_19 = params[46]
        alpha2_20 = params[47]
        alpha2_21 = params[48]
        alpha2_22 = params[49]
        alpha2_23 = params[50]
        alpha2_24 = params[51]
        alpha3_1 = params[52]
        alpha3_2 = params[53]
        alpha3_3 = params[54]
        alpha3_4 = params[55]
        alpha3_5 = params[56]
        alpha3_6 = params[57]
        alpha3_7 = params[58]
        alpha3_8 = params[59]
        alpha3_9 = params[60]
        alpha3_10 = params[61]
        alpha3_11 = params[62]
        alpha3_12 = params[63]
        alpha3_13 = params[64]
        alpha3_14 = params[65]
        alpha3_15 = params[66]
        alpha3_16 = params[67]
        alpha3_17 = params[68]
        alpha3_18 = params[69]
        alpha3_19 = params[70]
        alpha3_20 = params[71]
        alpha3_21 = params[72]
        alpha3_22 = params[73]
        alpha3_23 = params[74]
        alpha3_24 = params[75]

        gndis = []

        gndis.append(self.out(i, j, k, l, alpha1_1, alpha2_1, alpha3_1))
        gndis.append(self.out(i, j, l, k, alpha1_2, alpha2_2, alpha3_2))
        gndis.append(self.out(i, k, j, l, alpha1_3, alpha2_3, alpha3_3))
        gndis.append(self.out(i, k, l, j, alpha1_4, alpha2_4, alpha3_4))
        gndis.append(self.out(i, l, j, k, alpha1_5, alpha2_5, alpha3_5))
        gndis.append(self.out(i, l, k, j, alpha1_6, alpha2_6, alpha3_6))
        gndis.append(self.out(j, i, k, l, alpha1_7, alpha2_7, alpha3_7))
        gndis.append(self.out(j, i, l, k, alpha1_8, alpha2_8, alpha3_8))
        gndis.append(self.out(j, k, i, l, alpha1_9, alpha2_9, alpha3_9))
        gndis.append(self.out(j, k, l, i, alpha1_10, alpha2_10, alpha3_10))
        gndis.append(self.out(j, l, i, k, alpha1_11, alpha2_11, alpha3_11))
        gndis.append(self.out(j, l, k, i, alpha1_12, alpha2_12, alpha3_12))
        gndis.append(self.out(k, i, j, l, alpha1_13, alpha2_13, alpha3_13))
        gndis.append(self.out(k, i, l, j, alpha1_14, alpha2_14, alpha3_14))
        gndis.append(self.out(k, j, i, l, alpha1_15, alpha2_15, alpha3_15))
        gndis.append(self.out(k, j, l, i, alpha1_16, alpha2_16, alpha3_16))
        gndis.append(self.out(k, l, i, j, alpha1_17, alpha2_17, alpha3_17))
        gndis.append(self.out(k, l, j, i, alpha1_18, alpha2_18, alpha3_18))
        gndis.append(self.out(l, i, j, k, alpha1_19, alpha2_19, alpha3_19))
        gndis.append(self.out(l, i, k, j, alpha1_20, alpha2_20, alpha3_20))
        gndis.append(self.out(l, j, i, k, alpha1_21, alpha2_21, alpha3_21))
        gndis.append(self.out(l, j, k, i, alpha1_22, alpha2_22, alpha3_22))
        gndis.append(self.out(l, k, i, j, alpha1_23, alpha2_23, alpha3_23))
        gndis.append(self.out(l, k, j, i, alpha1_24, alpha2_24, alpha3_24))

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
        return ["i","j","k","l",
                "alpha1_1",
                "alpha1_2",
                "alpha1_3",
                "alpha1_4",
                "alpha1_5",
                "alpha1_6",
                "alpha1_7",
                "alpha1_8",
                "alpha1_9",
                "alpha1_10",
                "alpha1_11",
                "alpha1_12",
                "alpha1_13",
                "alpha1_14",
                "alpha1_15",
                "alpha1_16",
                "alpha1_17",
                "alpha1_18",
                "alpha1_19",
                "alpha1_20",
                "alpha1_21",
                "alpha1_22",
                "alpha1_23",
                "alpha1_24",
                "alpha2_1",
                "alpha2_2",
                "alpha2_3",
                "alpha2_4",
                "alpha2_5",
                "alpha2_6",
                "alpha2_7",
                "alpha2_8",
                "alpha2_9",
                "alpha2_10",
                "alpha2_11",
                "alpha2_12",
                "alpha2_13",
                "alpha2_14",
                "alpha2_15",
                "alpha2_16",
                "alpha2_17",
                "alpha2_18",
                "alpha2_19",
                "alpha2_20",
                "alpha2_21",
                "alpha2_22",
                "alpha2_23",
                "alpha2_24",
                "alpha3_1",
                "alpha3_2",
                "alpha3_3",
                "alpha3_4",
                "alpha3_5",
                "alpha3_6",
                "alpha3_7",
                "alpha3_8",
                "alpha3_9",
                "alpha3_10",
                "alpha3_11",
                "alpha3_12",
                "alpha3_13",
                "alpha3_14",
                "alpha3_15",
                "alpha3_16",
                "alpha3_17",
                "alpha3_18",
                "alpha3_19",
                "alpha3_20",
                "alpha3_21",
                "alpha3_22",
                "alpha3_23",
                "alpha3_24"
                ]
