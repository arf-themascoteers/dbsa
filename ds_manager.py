import os.path

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self,features=4200,samples=21782):
        np.random.seed(0)
        self.dataset_file_name = f"dataset_{features}_{samples}.csv"
        dataset = os.path.join("data", self.dataset_file_name)
        df = pd.read_csv(dataset)
        bands = len(df.columns) - 1
        self.X_columns = [str(i) for i in range(bands)]
        self.y_column = "oc"
        df = df[self.X_columns+[self.y_column]]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()
        self.full_data = DSManager._normalize(self.full_data)

    def __repr__(self):
        return self.dataset_file_name

    def count_rows(self):
        return self.full_data.shape[0]

    def count_features(self):
        return len(self.X_columns)

    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_X_y(self):
        return self.get_X_y_from_data(self.full_data)

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        return train_data, test_data

    def get_train_test_X_y(self):
        train_data, test_data = self.get_train_test()
        return *DSManager.get_X_y_from_data(train_data), \
            *DSManager.get_X_y_from_data(test_data)
