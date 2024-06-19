from module import dataset, Networks
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyvista as pv
import sys

train_list, test_list = dataset.train_test_split()
train_dataset = dataset.RelDistance(train_list)
test_dataset = dataset.RelDistance(test_list)
train_loader = DataLoader(train_dataset, batch_size = 8)
test_loader = DataLoader(test_dataset, batch_size = 8)

epochs = 1000
net = Networks.GCN_edge().to("cuda")
criterion = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr = 1e-3)

trainloss_history = []
testloss_history = []
best_state = None
best_loss = float("inf")

for epoch in range(epochs):
    loss_sum = 0.
    count = 0
    for train_data in train_loader:
        preds = net(train_data)
        loss = criterion(preds, train_data.y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
        count += 1
    trainloss_history.append(loss_sum/count)

    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for test_data in test_loader:
            preds = net(test_data)
            loss = criterion(preds, test_data.y)
            loss_sum += loss.item()
            count += 1
        testloss_history.append(loss_sum/count)

        if testloss_history[-1] < best_loss:
            torch.save(net.state_dict(), "weight.pth")
            best_loss = testloss_history[-1]
            
    print("epoch = {0} : train_loss = {1}, test_loss = {2}".format(epoch, trainloss_history[-1], testloss_history[-1]))

trainloss_history = np.array(trainloss_history)
testloss_history = np.array(testloss_history)
loss_history = np.stack((
    np.arange(epochs),
    trainloss_history,
    testloss_history
), axis = 1)

loss_history = pd.DataFrame(loss_history)
loss_history.to_csv("loss_history.csv")