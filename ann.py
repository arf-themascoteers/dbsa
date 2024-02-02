import torch.nn as nn
import torch
import my_utils
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, sis, lock):
        super().__init__()
        self.lock = lock
        self.device = my_utils.get_device()
        modules = []
        self.count_sis = 0
        for si in sis:
            si_class = si["si"]
            number_of_instances = si["count"]
            self.count_sis = self.count_sis + number_of_instances
            initial_values = None
            if "initial_values" in si:
                initial_values = si["initial_values"]
            for i in range(number_of_instances):
                if initial_values is None:
                    module = si_class()
                else:
                    initial_value = initial_values[i]
                    module = si_class(initial_value)
                modules.append(module)

        self.si_modules = nn.ModuleList(modules)
        self.linear = nn.Sequential(
            nn.Linear(self.count_sis, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 1)
        )
        for module in self.si_modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = torch.zeros(size, self.count_sis, dtype=torch.float32).to(self.device)
        for i,module in enumerate(self.si_modules):
            outputs[:,i] = module(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_params(self, scale_index=1):
        params = {}
        si_count = {}
        for module in self.si_modules:
            name = str(module)
            if name not in si_count:
                si_count[name] = 0
            else:
                si_count[name] = si_count[name] + 1
            name_serial = f"{name}_{si_count[name]}"
            names = module.names()
            for i in range(module.count_params):
                param_name = f"{name_serial}:{names[i]}"
                param_value = module.get_param_value(i)
                if i < module.count_indices:
                    param_value = F.sigmoid(torch.tensor(param_value)).item()
                    param_value = param_value*scale_index
                    if scale_index != 1:
                        param_value = round(param_value)
                    else:
                        param_value = round(param_value,5)
                else:
                    param_value = round(param_value, 5)
                params[param_name]=param_value
        return params

    def get_param_names(self):
        return list(self.get_params().keys())
