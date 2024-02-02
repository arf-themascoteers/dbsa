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
            {"si": MNDI, "count": 1, "initial_values":[torch.tensor([0.171, 0.714, 0.948, 1.5])], "lock":True}
            #{"si": MNDI, "count": 1}
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