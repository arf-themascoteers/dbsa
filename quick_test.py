from eval_ann import EvalANN
from ds_manager import DSManager
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


feature = 4200
sample = 21782
bands = [400, 2991, 3928]

dataset = DSManager(feature, sample)
X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
print(X_train)





