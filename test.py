from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": BI, "count": 5}
        ],
        [
            {"si": DI, "count": 5}
        ],
        [
            {"si": RI, "count": 5}
        ],
        [
            {"si": NDI, "count": 5}
        ],
        [
            {"si": SNDI, "count": 5}
        ],
        [
            {"si": MNDI, "count": 5}
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