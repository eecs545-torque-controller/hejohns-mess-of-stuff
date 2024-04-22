#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas
import numpy as np
import json

#with open('./total.json', 'r') as f:
#    data = json.load(f);
#print(type(data))
#for subject, subject_data in data:
#    for activity in subject_data:
#        print(f"{subject}/{activity}")

df = pandas.read_csv('./airline-passengers.csv')
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
timeseries = df[["Passengers"]].values.astype('float32')
#plt.plot(timeseries)
#plt.show()
train_size = int(len(timeseries) * 0.5)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        feature = dataset[i : i + window_size]
        target = dataset[i + 1 : i + window_size + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

lookback = 20
X_train, y_train = create_dataset(train, window_size=lookback)
X_test, y_test = create_dataset(test, window_size=lookback)
print(len(train), len(test))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x[:, -1, :]
        x = self.linear(x)
        return x

model = AirModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

nepochs = 2000
for epoch in range(nepochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = model(X_train)[:, -1, :]

    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size + lookback : len(timeseries)] = model(X_test)[:, -1, :]

plt.plot(timeseries, c='b')
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
