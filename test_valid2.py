from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
from p_mndi import P_MNDI
import torch

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": BI, "count": 1, "initial_values":[torch.tensor([0.2840])]},
            {"si": BI, "count": 1, "initial_values":[torch.tensor([0.3635])]},
            {"si": BI, "count": 1, "initial_values":[torch.tensor([0.6471])]}
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