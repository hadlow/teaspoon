import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

n_locations = 10
batch_size = 30
epochs = 5
losses = []

class Model(nn.Module):
    def __init__(self, n_locations, h1 = 100, h2 = 100):
        super().__init__()

        self.fc1 = nn.Linear(n_locations * 2, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, n_locations)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

model = Model(n_locations)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

x_train = torch.FloatTensor(pd.read_csv('./data/x_train.csv').values)
x_test = torch.FloatTensor(pd.read_csv('./data/x_test.csv').values)

y_train_raw = pd.read_csv('./data/y_train.csv')
y_train = torch.FloatTensor(y_train_raw.drop('11', axis = 1).values)
y_train_tour_lengths = y_train_raw['11'].values

y_test_raw = pd.read_csv('./data/y_test.csv')
y_test = torch.FloatTensor(y_test_raw.drop('11', axis = 1).values)
y_test_tour_lengths = y_test_raw['11'].values

for epoch in range(epochs):
    y_pred = model.forward(x_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss)

    print(f'Epoch {epoch}: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.xlabel('Loss')
plt.ylabel('Epoch')
