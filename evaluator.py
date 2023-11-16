from ds_manager import DSManager
from sklearn.metrics import mean_squared_error, r2_score
import math
from ann_vanilla import ANNVanilla
from plott import plot_me_plz


class Evaluator:
    def __init__(self, dwt=False):
        self.dwt = dwt

    def process(self,dwt=True,indexify="sigmoid", retain_relative_position=True,random_initialize=True):
        ds = DSManager(dwt)
        train_x, train_y, test_x, test_y, validation_x, validation_y = ds.get_datasets()
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        ann = ANNVanilla(train_x, train_y, test_x, test_y, validation_x, validation_y,
                         dwt,indexify, retain_relative_position,random_initialize)
        ann.train()
        y_hats = ann.test()
        r2 = r2_score(test_y, y_hats)
        rmse = math.sqrt(mean_squared_error(test_y, y_hats, squared=False))
        plot_me_plz(dwt, indexify, retain_relative_position, random_initialize)
        return max(r2,0), rmse

