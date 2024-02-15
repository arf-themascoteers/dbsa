import torch.nn as nn
import torch
import my_utils
import torch.nn.functional as F
from approximator import get_splines


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
        self.linear_input_length = self._count_linear_input()
        self.linear = nn.Sequential(
            nn.Linear(self.linear_input_length, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 1)
        )
        if self.lock:
            for module in self.si_modules:
                for param in module.parameters():
                    param.requires_grad = False

    def _count_linear_input(self):
        counter = 0
        for i,module in enumerate(self.si_modules):
            counter = counter + module.get_output_length()
        return counter

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = []#torch.zeros(size, self.count_sis, dtype=torch.float32).to(self.device)
        for i,module in enumerate(self.si_modules):
            outputs.append(module(spline))
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.to(self.device)
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

    def generate_impact(self, X_train, y_train):
        criterion = torch.nn.MSELoss(reduction='mean')
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        spline = get_splines(X_train, self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_hat = self(spline)
        loss = criterion(y_hat, y_train)
        loss.backward()
        return self.get_impacts()

    def get_impacts(self):
        names = []
        grads = []

        for module in self.si_modules:
            module_name = module.__class__.__name__
            this_grads = module.params.grad
            for i in range(module.count_params):
                names.append(f"{module_name}-{module.names()[i]}")
                grads.append(torch.abs(this_grads[i]).item())

        sum_grads = sum(grads)
        grads = [g / sum_grads for g in grads]
        return names, grads