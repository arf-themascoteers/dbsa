import torch.nn as nn
import torch
import my_utils


class ANN(nn.Module):
    def __init__(self, sis):
        super().__init__()
        self.device = my_utils.get_device()
        modules = []
        self.input_size = 0
        for si in range(sis):
            si_class = si["si"]
            number_of_instances = si["count"]
            initial_values = si["initial_values"]
            for i in range(number_of_instances):
                initial_value = initial_values[i]
                module = si_class(initial_value)
                modules.append(module)
                self.input_size = self.input_size + module.count_indices

        self.modules = nn.ModuleList(modules)
        self.linear = nn.Sequential(
            nn.Linear(self.input_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,module in enumerate(self.modules):
            outputs[:,i] = module(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_params(self):
        last_name = None
        params = {}
        serial = 0
        for module in self.modules:
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
                params[param_name]=param_value
        return params

    def get_param_names(self):
        return list(self.get_params().keys())
