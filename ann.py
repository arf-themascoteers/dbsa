import torch.nn as nn
import torch
from algorithms.fsdr.band_index import BandIndex
import my_utils


class ANN(nn.Module):
    def __init__(self, target_feature_size, original_feature_size, seq, mode):
        super().__init__()
        self.device = my_utils.get_device()
        self.target_feature_size = target_feature_size
        self.original_feature_size = original_feature_size
        self.seq = seq
        self.mode = mode
        init_vals = torch.linspace(0.001,0.99, target_feature_size+2)
        modules = []
        for i in range(self.target_feature_size):
            val = my_utils.inverse_sigmoid_torch(init_vals[i + 1])
            bi = BandIndex(val, self.original_feature_size, self.seq, self.mode)
            modules.append(bi)
        self.band_indices = nn.ModuleList(modules)
        self.linear = self.get_linear()

    def get_linear(self):
        input_size = self.target_feature_size
        hidden_1 = 15
        hidden_2 = 10
        if self.seq:
            if self.mode == "linear_multi":
                input_size = sum([band_index.get_embedding_size() for band_index in self.band_indices])
        return nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_2, 1)
        )

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