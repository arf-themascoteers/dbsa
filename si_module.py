import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class SIModule(nn.Module):
    def __init__(self, count_params, count_indices, initial_value=None):
        super().__init__()
        self.count_params = count_params
        self.count_indices = count_indices
        if initial_value is None:
            initial_value = torch.rand(count_params)
        for i in range(self.count_indices):
            initial_value[i] = self.inverse_sigmoid(initial_value[i])
        self.params = nn.Parameter(initial_value)

    def inverse_sigmoid(self, x):
        return math.log(x / (1 - x))

    def forward(self, splines):
        indices = self.params[0:self.count_indices]
        indices = F.sigmoid(indices)
        outs1 = splines(indices)
        outs2 = self.params[self.count_indices:]
        outs = torch.hstack((outs1, outs2))
        outs = self._forward(outs)
        return outs

    def _forward(self, spline):
        pass

    def _names(self):
        pass
