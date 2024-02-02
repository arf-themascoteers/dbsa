from eval_ann import EvalANN
from ds_manager import DSManager
from sklearn.metrics import r2_score
import torch
import my_utils


feature = 4200
sample = 21782
bands = [720, 3000, 3980, 500, 1000, 1500, 2000, 4200]

dataset = DSManager(feature, sample)
X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
X_train = X_train[:,[bands]]
X_test = X_test[:,[bands]]

device = my_utils.get_device()
model = EvalANN(len(bands))
model.train()
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

lr = 0.001
weight_decay = lr / 10
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
device = my_utils.get_device()
model.to(device)
criterion = torch.nn.MSELoss(reduction='mean')
for epoch in range(1500):
    y_hat = model(X_train)
    loss = criterion(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())

model.eval()
y_hat = model(X_test)
loss = criterion(y_hat, y_test)
print(loss.item())
y_hat = y_hat.detach().cpu().numpy()
y_test = y_test.detach().cpu().numpy()
r2 = r2_score(y_test, y_hat)
print(r2)





