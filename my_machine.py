import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from approximator import get_splines
import torch
from ann import ANN
from datetime import datetime
import os
import my_utils


class MyMachine:
    def __init__(self, sis):
        self.model = ANN(sis)
        self.lr = 0.001
        self.weight_decay = self.lr/10
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.device = my_utils.get_device()
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.epochs = 1500
        self.csv_file = os.path.join("results", f"{str(datetime.now().timestamp()).replace('.', '')}.csv")
        self.start_time = datetime.now()
        print("Learnable Params",sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def score(self, X_train, y_train, X_test, y_test):
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1,random_state=40)
        self.fit(X_train, X_validation, y_train, y_validation)
        r2_train, rmse_train = self.evaluate(X_train, y_train)
        r2_validation, rmse_validation = self.evaluate(X_validation, y_validation)
        r2_test, rmse_test = self.evaluate(X_test, y_test)
        return r2_train, r2_validation, r2_test, rmse_train, rmse_validation, rmse_test

    def fit(self, X_train, X_validation, y_train, y_validation):
        print(f"X_train,X_validation: {X_train.shape} {X_validation.shape}")
        self.write_columns()
        self.model.train()
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        spline = get_splines(X_train, self.device)
        X_validation = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
        spline_validation = get_splines(X_validation, self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_validation = torch.tensor(y_validation, dtype=torch.float32).to(self.device)
        for epoch in range(self.epochs):
            y_hat = self.model(spline)
            loss_1 = self.criterion(y_hat, y_train)
            loss_2 = 0
            loss = loss_1 + loss_2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            row = self.dump_row(epoch, X_train, spline, y_train, X_validation, spline_validation, y_validation)
            if epoch%50 == 0:
                print("".join([str(i).ljust(20) for i in row]))

    def evaluate(self,X,y,spline=None):
        self.model.eval()
        if spline is None:
            spline = get_splines(X, self.device)
        y_hat = self.model(spline)
        y_hat = y_hat.reshape(-1)
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        r2 = r2_score(y, y_hat)
        rmse = math.sqrt(mean_squared_error(y, y_hat))
        self.model.train()
        return max(r2,0), rmse

    def write_columns(self):
        columns = ["epoch","train_r2","validation_r2","train_rmse","validation_rmse"]
        names = self.model.get_param_names()
        columns = columns + names
        print("".join([c.ljust(20) for c in columns]))
        with open(self.csv_file, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def dump_row(self, epoch, X, spline, y, X_validation, spline_validation, y_validation):
        train_r2, train_rmse = self.evaluate(X, y, spline)
        validation_r2, validation_rmse = self.evaluate(X_validation, y_validation, spline_validation)
        row = [train_r2, validation_r2, train_rmse, validation_rmse]
        row = [round(r,5) for r in row]
        row = [epoch] + row
        for key, value in self.model.get_params(X.shape[1]):
            print(f"{value.ljust(20)}")
        with open(self.csv_file, 'a') as file:
            file.write(",".join([f"{x}" for x in row]))
            file.write("\n")
        return row
