from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
from gndi2 import GNDI2
from gndi3 import GNDI3
from gndi4 import GNDI4
from gndi3alpha import GNDI3Alpha
from gndi2alpha import GNDI2Alpha
import torch

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": GNDI3Alpha, "count": 1}
        ],
        [
            {"si": GNDI3, "count": 1}
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