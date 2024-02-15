from evaluator import Evaluator
from gndi2 import GNDI2
import torch

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
        ],
        [
            {"si": GNDI2, "count": 1}
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