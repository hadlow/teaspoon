import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from python_tsp.distances import euclidean_distance_matrix

from encoder_decoder import decode

n_locations = 10
batch_size = 64
train_examples = 20
epochs = 400
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
        x = torch.sigmoid(self.output(x))

        return x

model = Model(n_locations)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

x = pd.read_csv('./data/x_raw.csv')
y = pd.read_csv('./data/y_encoded.csv')

train_dataset = TensorDataset(torch.FloatTensor(x.values[:train_examples]), torch.FloatTensor(np.delete(y.values[:train_examples], 10, axis=1)))
test_dataset = TensorDataset(torch.FloatTensor(x.values[train_examples:]), torch.FloatTensor(np.delete(y.values[train_examples:], 10, axis=1)))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch in train_loader:
        x_train = batch[0]
        y_train = batch[1] 

        y_pred = model.forward(x_train)

        loss = criterion(y_pred, y_train)

        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}: {loss}')

plt.plot(range(epochs), losses)
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.show()

def draw_edge(a, b):
    x = [a[0], b[0]]
    y = [a[1], b[1]]

    plt.plot(x, y, 'k-')

def draw_route(permutation, points):
    for position_index, location_index in enumerate(permutation):
        if len(permutation) == position_index + 1:
            next_location_index = permutation[0]
        else:
            next_location_index = permutation[position_index + 1]

        location_from = points[location_index]
        location_to = points[next_location_index]
        
        draw_edge(location_from, location_to)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

for index, test_batch in enumerate(test_loader):
    if index > 0:
        break

    random_sample = random.randint(0, 10)
    test_x = batch[0][random_sample]

    pred = model.forward(test_x)
    decoded_pred = decode(pred.detach().numpy()).astype('int')

    test_x = test_x.detach().numpy().reshape(-1, 2)

    draw_route(decoded_pred, test_x)

