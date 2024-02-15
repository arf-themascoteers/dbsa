from evaluator import Evaluator
from gndi2 import GNDI2
from gndi3 import GNDI3
from sndi import SNDI
from mndi import MNDI
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
    names, grads = model.generate_impact_of_sis_ind(train_x, train_y)
    for i in range(len(names)):
        print(f"{names[i]}\t{grads[i]}")


if __name__ == '__main__':
    feature = 4200
    sample = 21782
    sis = [
        {"si": SNDI, "count": 2},
        {"si": MNDI, "count": 3},
        #{"si": GNDI2, "count": 1},
    ]
    dataset = DSManager(feature, sample)
    train_x, train_y, test_x, test_y = dataset.get_train_test_X_y()

    #model_path = "models/1707992241887198.pth"
    model_path = train_now(feature, sample, sis)
    analyze(model_path, train_x, train_y)


