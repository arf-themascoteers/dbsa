from sklearn.linear_model import LinearRegression
import torch
import math


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
