from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
import torch

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": SNDI, "count": 1, "initial_values":[torch.tensor([3745/4200, 962/4200, 1.01024])], "lock":True},
        ],
        [
            {"si": MNDI, "count": 1, "initial_values": [torch.tensor([3822 / 4200, 529 / 4200, 2966 / 4200, 0.91504])], "lock": True},
        ]
    ]

    tasks = []
    for feature in features:
        for sample in samples:
            for sis in siss:
                tasks.append(
                    {
                        "feature": feature,
                        "sample": sample,
                        "sis": sis
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()