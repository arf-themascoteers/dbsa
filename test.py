from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
import torch

if __name__ == '__main__':
    features = [66]
    samples = [871]
    siss = [
        [
            {"si": BI, "count": 3},
            {"si": NDI, "count": 3},
            {"si": BI, "count": 2, "initial_values" : [torch.tensor([0.01]), torch.tensor([0.99])]},
            {"si": SNDI, "count": 2, "initial_values" : [torch.tensor([0.01, 0.05, 0.6]), torch.tensor([0.9, 0.3, 3.0])]}
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
                        "sis": sis,
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()