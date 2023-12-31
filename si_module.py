import torch.nn as nn
import torch
import torch.nn.functional as F


class SIModule(nn.Module):
    def __init__(self, count_params, initial_value=None):
        super().__init__()
        iv = initial_value
        if iv is None:
            iv = (torch.rand(count_params)*10)-5
        self.params = nn.Parameter(iv)

    def forward(self, spline):
        outs = [spline.evaluate(F.sigmoid(param)) for param in self.params]
        outs = self._forward(outs)
        return outs

    def _forward(self, spline):
        pass

    def _names(self):
        pass

    def param_values(self):
        return [{"name":self._names()[i],"value":self.indexify(i)} for i in range(self.params.shape[0])]

    def indexify(self, p):
        return round((F.sigmoid(self.params[p])).item()*4200)