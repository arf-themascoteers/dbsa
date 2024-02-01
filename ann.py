import torch.nn as nn
import torch
import my_utils


class ANN(nn.Module):
    def __init__(self, sis):
        super().__init__()
        self.device = my_utils.get_device()
        modules = []
        for si in range(sis):
            si_class = si["si"]
            si_count = si["count"]
            for i in range(si_count):
                modules.append(si_class())

        self.band_indices = nn.ModuleList(modules)
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_2, 1)
        )

    def get_linear(self):
        input_size = self.target_feature_size
        hidden_1 = 15
        hidden_2 = 10
        return

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,band_index in enumerate(self.band_indices):
            outputs[:,i] = band_index(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return [band_index.get_indices() for band_index in self.band_indices]

    def get_flattened_indices(self):
        indices = torch.cat((self.get_indices()), dim=0)
        return indices.tolist()