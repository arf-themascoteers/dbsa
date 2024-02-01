from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI

if __name__ == '__main__':
    features = [66]
    samples = [871]
    siss = [
        [
            {"si": DI, "count": 2},
            {"si": NDI, "count": 2},
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