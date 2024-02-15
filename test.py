from evaluator import Evaluator
from bi import BI
from di import DI
from ri import RI
from ndi import NDI
from sndi import SNDI
from mndi import MNDI
from gndi_2 import GNDI_2
from gndi_3 import GNDI_3
from gndi_4 import GNDI_4
import torch

if __name__ == '__main__':
    features = [4200]
    samples = [21782]
    siss = [
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],
        [
            {"si": GNDI_2, "count": 1}
        ],



        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],
        [
            {"si": GNDI_3, "count": 1}
        ],

        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
        ],
        [
            {"si": GNDI_4, "count": 1}
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