import torch
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


def get_splines(X, device):
    X = X.permute(1, 0)
    indices = torch.linspace(0, 1, X.shape[0]).to(device)
    coeffs = natural_cubic_spline_coeffs(indices, X)
    spline = NaturalCubicSpline(coeffs)
    return spline


if __name__ == "__main__":
    x = torch.rand((10,20))
    s = get_splines(x, "cpu")
    a = s.evaluate(torch.tensor([0.1,0.2]))
    print(a)