import math
from sklearn.metrics import mean_squared_error, r2_score
from spectral_dataset import SpectralDataset
import torch
from torch.utils.data import DataLoader
from ann import ANN
from reporter import Reporter
from datetime import datetime


class ANNVanilla:
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y, dwt=True,indexify="sigmoid", retain_relative_position=True,random_initialize=True):
        print(f"dwt={dwt},indexify={indexify}, retain_relative_position={retain_relative_position},random_initialize={random_initialize}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = SpectralDataset(train_x, train_y)
        self.test_dataset = SpectralDataset(test_x, test_y)
        self.validation_dataset = SpectralDataset(validation_x, validation_y)
        self.feature_size = validation_x.shape[1]
        self.retain_relative_position = retain_relative_position
        self.model = ANN(self.feature_size, random_initialize,indexify="sigmoid")
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.epochs = 1000
        self.batch_size = 1000
        self.dwt = dwt
        if not self.dwt:
            self.epochs = 400
        self.model_name = f"{str(dwt)}_{indexify}_{str(retain_relative_position)}_{str(random_initialize)}.pt"
        self.start_time = datetime.now()

    def get_elapsed_time(self):
        return datetime.now() - self.start_time

    def train(self):
        self.write_columns()
        self.model.train()
        px = []
        for params in self.model.machines:
            for p in params.parameters():
                px.append(p)

        param_group1 = {'params': px, 'lr': 0.01, "betas":(0.9, 0.999)}
        param_group2 = {'params': self.model.linear1.parameters(), 'lr': 0.001}
        optimizer = torch.optim.Adam([param_group1,param_group2], lr=0.01, weight_decay=0.001)
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
            print("".join([str(i).ljust(10) for i in row]))
            Reporter.write_row(rows)
            torch.save(self.model, self.model_name)

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
        columns = ["epoch","train_r2","train_rmse","test_r2","test_rmse","validation_r2","validation_rmse","time"]
        for index,p in enumerate(self.model.get_indices()):
            columns.append(f"band_{index}")
        print("".join([c.ljust(10) for c in columns]))
        Reporter.write_columns(columns)

    def dump_row(self, epoch):
        train_r2, train_rmse = self.train_results()
        test_r2, test_rmse = self.test_results()
        validation_r2, validation_rmse = self.validation_results()
        plot_items = [epoch, train_r2, train_rmse, test_r2, test_rmse, validation_r2, validation_rmse, self.get_elapsed_time()]
        for p in self.model.get_indices():
            plot_items.append(ANNVanilla.indexify_raw_index(p))
        Reporter.write_row(plot_items)
        return plot_items

    @staticmethod
    def indexify_raw_index(raw_index):
        return round(raw_index * self.feature_size,5)
