import torch.nn as nn
import torch
import torch.nn.functional as F
from band_index import BandIndex
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


class ANN(nn.Module):
    def __init__(self, spline_indices, random_initialize=True,indexify="sigmoid"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_band_count = 10
        self.indexify = indexify
        self.initial_values = torch.linspace(0.05, 0.95, 10)
        if self.indexify == "sigmoid":
            self.initial_values = ANN.inverse_sigmoid_torch(self.initial_values)

        self.linear1 = nn.Sequential(
            nn.Linear(self.target_band_count, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        self.indices = torch.linspace(0, 1, spline_indices).to(self.device)
        modules = []
        for i in range(self.target_band_count):
            initial_value = None
            if not random_initialize:
                initial_value = self.initial_values[i]
            modules.append(BandIndex(initial_value=initial_value))
        self.machines = nn.ModuleList(modules)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, x):
        outputs = torch.zeros(x.shape[0], self.target_band_count, dtype=torch.float32).to(self.device)
        x = x.permute(1,0)
        coeffs = natural_cubic_spline_coeffs(self.indices, x)
        spline = NaturalCubicSpline(coeffs)

        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)

        soc_hat = self.linear1(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def retention_loss(self):
        loss = None
        for i in range(1, len(self.machines)):
            later_band = self.machines[i].raw_index
            past_band = self.machines[i-1].raw_index
            this_loss = F.relu(past_band-later_band)
            if loss is None:
                loss = this_loss
            else:
                loss = loss + this_loss
        return loss

    def get_indices(self):
        return [machine.index_value() for machine in self.machines]
