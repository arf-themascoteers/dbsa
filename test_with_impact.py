from evaluator import Evaluator
from gndi_2 import GNDI_2
import torch
from ds_manager import DSManager


def train_now(feature, sample, sis):
    task = {
        "feature": feature,
        "sample": sample,
        "sis": sis
    }

    ev = Evaluator([task])
    model_files = ev.evaluate()
    return model_files[0]


def analyze(model_path, train_x, train_y):
    model = torch.load(model_path)
    names, grads = model.generate_impact(train_x, train_y)
    for i in range(len(names)):
        print(f"{names[i]}\t{grads[i]}")


if __name__ == '__main__':
    feature = 4200
    sample = 21782
    sis = [
        {"si": GNDI_2, "count": 1}
    ]
    dataset = DSManager(feature, sample)
    train_x, train_y, test_x, test_y = dataset.get_train_test_X_y()

    #model_path = "models/1707990172214642.pth"
    model_path = train_now(feature, sample, sis)
    analyze(model_path, train_x, train_y)


