from sklearn.linear_model import LinearRegression
import torch


def inverse_sigmoid_torch(x):
    return -torch.log(1.0 / x - 1.0)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
