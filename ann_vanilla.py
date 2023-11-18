import math
from sklearn.metrics import mean_squared_error, r2_score
import plott
from spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
from ann import ANN
from reporter import Reporter
from datetime import datetime
import os

class ANNVanilla:
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y,
                 dwt=True,indexify="sigmoid", retain_relative_position=True,random_initialize=True,uniform_lr=True, lr=0.0001, skip=True):
        print(f"dwt={dwt},indexify={indexify}, retain_relative_position={retain_relative_position},random_initialize={random_initialize},uniform_lr={uniform_lr}")
        self.skip = skip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)
        self.uniform_lr = uniform_lr
        self.lr = lr
        self.feature_size = validation_x.shape[1]
        self.retain_relative_position = retain_relative_position
        self.model = ANN(self.feature_size, random_initialize,indexify="sigmoid",skip=self.skip)
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.epochs = 2500
        self.batch_size = 1000
        self.dwt = dwt
        if not self.dwt:
            self.epochs = 2500
        self.prefix = f"{str(dwt)}_" \
                      f"{indexify}_" \
                      f"{str(retain_relative_position)}_" \
                      f"{str(random_initialize)}_" \
                      f"{str(self.uniform_lr)}_" \
                      f"{str(self.lr)}_" \
                      f"{str(self.skip)}"
        self.csv_file = self.prefix+".csv"
        self.done_file = self.prefix+".txt"
        self.model_file = self.prefix+".pt"
        self.csv_file = os.path.join("results",self.csv_file)
        self.done_file = os.path.join("results",self.done_file)
        self.model_file = os.path.join("results",self.model_file)
        self.reporter = Reporter(self.csv_file)
        self.start_time = datetime.now()

    def get_elapsed_time(self):
        return (datetime.now() - self.start_time).total_seconds()

    def create_optimizer(self):
        weight_decay = self.lr/10
        if self.uniform_lr:
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        special_lr = self.lr*3
        param_group1 = {'params': self.model.machines.parameters(), 'lr': special_lr, "betas":(0.8, 0.888)}
        param_group2 = {'params': self.model.linear1.parameters(), 'lr': self.lr}
        param_group3 = {'params': self.model.linear2.parameters(), 'lr': self.lr}
        return torch.optim.Adam([param_group1,param_group2,param_group3], lr=self.lr, weight_decay=weight_decay)

    def train(self):
        if os.path.exists(self.done_file):
            print(f"Already done {self.done_file}. Skipping.")
            return
        self.write_columns()
        self.model.train()
        optimizer = self.create_optimizer()
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            rows = []
            for batch_number,(x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                if self.retain_relative_position:
                    loss = loss + self.model.retention_loss()
                for machine in self.model.machines:
                    loss = loss + machine.range_loss()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            row = self.dump_row(epoch)
            rows.append(row)
            print("".join([str(i).ljust(20) for i in row]))
            torch.save(self.model, self.model_file)
        plott.plot_me_plz(self.csv_file)

        with open(self.done_file, 'w') as file:
            file.write(str(datetime.now()))

    def evaluate(self,dataset):
        batch_size = 30000
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            y_hat = y_hat.reshape(-1)
            y_hat = y_hat.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            r2 = r2_score(y, y_hat)
            rmse = math.sqrt(mean_squared_error(y, y_hat))
            self.model.train()
            return max(r2,0), rmse

    def train_results(self):
        return self.evaluate(self.train_dataset)

    def test_results(self):
        return self.evaluate(self.test_dataset)

    def validation_results(self):
        return self.evaluate(self.validation_dataset)

    def write_columns(self):
        columns = ["epoch","train_r2","test_r2","validation_r2","train_rmse","test_rmse","validation_rmse","time"]
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index+1}")
        print("".join([c.ljust(20) for c in columns]))
        self.reporter.write_columns(columns)

    def dump_row(self, epoch):
        train_r2, train_rmse = self.train_results()
        test_r2, test_rmse = self.test_results()
        validation_r2, validation_rmse = self.validation_results()
        row = [train_r2, test_r2, validation_r2, train_rmse,test_rmse, validation_rmse]
        row = [round(r,5) for r in row]
        row = [epoch] + row + [self.get_elapsed_time()]
        for p in self.model.get_indices():
            row.append(self.indexify_raw_index(p))

        self.reporter.write_row(row)
        return row

    def indexify_raw_index(self, raw_index):
        return round(raw_index.item() * self.feature_size)
