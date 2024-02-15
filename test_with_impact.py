from evaluator import Evaluator
from gndi_2 import GNDI_2
import torch
import os
from ds_manager import DSManager
from ann import ANN


def train_now(feature, sample, sis, machine_name):
    task = {
        "feature": feature,
        "sample": sample,
        "sis": sis
    }

    ev = Evaluator([task])
    machine = ev.evaluate()
    torch.save(model, machine_name)


if __name__ == '__main__':
    feature = 4200
    sample = 21782
    sis = [
        {"si": GNDI_2, "count": 1}
    ]

    machine_name = "machine.pth"
    model = ANN(sis, False)
    if not os.path.exists(machine_name):
        train_now(feature, sample, sis, machine_name)
    model = torch.load(machine_name)
    dataset = DSManager(feature, sample)
    train_x, train_y, test_x, test_y = dataset.get_train_test_X_y()
    model.generate_impact(train_x, train_y)
