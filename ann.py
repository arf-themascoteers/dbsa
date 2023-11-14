import torch.nn as nn
import torch
from bi import BI
import torch.nn.functional as F

from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sis = [
            {"si":BI, "count":10, "initial_values": torch.tensor([-4.19722458, -1.38629436, -0.84729786, -0.40546511,
                                                                  0.40546511,  0.84729786,  1.38629436,  2.19722458,
                                                                  3.04452244,  6.60517019]).reshape(-1,1) }
        ]

        self.total = sum([si["count"] for si in self.sis])

        self.linear1 = nn.Sequential(
            nn.Linear(self.total, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )

        self.indices = torch.linspace(0, 1, 66).to(self.device)
        modules = []
        for si in self.sis:
            if "initial_values" in si:
                for i in range(si["count"]):
                    modules.append(si["si"](si["initial_values"][i]))
            else:
                modules = modules + [si["si"]() for i in range(si["count"])]
        self.machines = nn.ModuleList(modules)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.total, dtype=torch.float32).to(self.device)
        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)

        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)

        soc_hat = self.linear1(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_param_loss(self):
        loss = None
        for i in range(1, len(self.machines)):
            later_band = self.machines[i].params
            past_band = self.machines[i-1].params
            this_loss = F.relu(past_band-later_band)
            if loss is None:
                loss = this_loss
            else:
                loss = loss + this_loss
        return loss

    def get_params(self):
        params = []
        index = 0
        for type_count, si in enumerate(self.sis):
            for i in range(si["count"]):
                machine = self.machines[index]
                p = {}
                p["si"] = si["si"].__name__
                p["params"] = machine.param_values()
                params.append(p)
                index = index+1
        return params
