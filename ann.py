import torch.nn as nn
import torch
import my_utils


class ANN(nn.Module):
    def __init__(self, sis):
        super().__init__()
        self.device = my_utils.get_device()
        modules = []
        self.input_size = 0
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
                self.input_size = self.input_size + module.count_indices

        self.si_modules = nn.ModuleList(modules)
        self.linear = nn.Sequential(
            nn.Linear(self.input_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = torch.zeros(size, self.count_sis, dtype=torch.float32).to(self.device)
        for i,module in enumerate(self.si_modules):
            outputs[:,i] = module(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_params(self, scale_index=1):
        last_name = None
        params = {}
        serial = 0
        for module in self.si_modules:
            name = str(module)
            if last_name is None or name!=last_name:
                serial = 0
            else:
                serial = serial + 1
            last_name = name
            name_serial = f"{name}_{serial}"
            names = module.names()
            for i in range(module.count_params):
                param_name = f"{name_serial}:{names[i]}"
                param_value = module.get_param_value(i)
                if i < module.count_indices:
                    param_value = param_value*scale_index
                    if scale_index != 1:
                        param_value = round(param_value)
                params[param_name]=param_value
        return params

    def get_param_names(self):
        return list(self.get_params().keys())
