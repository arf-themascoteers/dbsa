import torch.nn as nn
import torch
import my_utils
import torch.nn.functional as F


class EvalANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.device = my_utils.get_device()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 1)
        )

    def forward(self, X):
        soc_hat = self.linear(X)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat
