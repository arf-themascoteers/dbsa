import torch.nn as nn
import torch
import torch.nn.functional as F
from my_utils import inverse_sigmoid


class SIModule(nn.Module):
    def __init__(self, count_params, count_indices, output_length, initial_value=None):
        super().__init__()
        self.count_params = count_params
        self.count_indices = count_indices
        self.output_length = output_length
        if initial_value is None:
            initial_value = torch.rand(count_params)
        else:
            if count_params != initial_value.shape[0]:
                raise TypeError(f"{self.__class__.__name__} requires {self.count_params} arguments. "
                                f"Given {initial_value.shape[0]} ({initial_value}).")
        for i in range(self.count_indices):
            initial_value[i] = inverse_sigmoid(initial_value[i])
        self.params = nn.Parameter(initial_value)

    def get_output_length(self):
        return self.output_length

    def forward(self, splines):
        outs1 = self.params[0:self.count_indices]
        outs1 = F.sigmoid(outs1)
        if self.count_indices != self.count_params:
            outs2 = self.params[self.count_indices:]
            outs1 = torch.hstack((outs1, outs2))
        return self._forward(splines, outs1)

    def _forward(self, splines, params):
        pass

    def _names(self):
        pass

    def names(self):
        return self._names()

    def get_param_value(self, index):
        return self.params[index].item()

    def __str__(self):
        return str(self.__class__.__name__)
