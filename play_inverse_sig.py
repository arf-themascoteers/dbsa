import torch


def inverse_sigmoid_torch(x):
    return -torch.log(1.0 / x - 1.0)

x = torch.linspace(0.05,0.95,10)
result = inverse_sigmoid_torch(x)
print(result)