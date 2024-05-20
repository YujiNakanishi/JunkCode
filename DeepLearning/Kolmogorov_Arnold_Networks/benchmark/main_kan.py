from KAN import KANLayer
from dataset import *
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np

neuron = 8
base_num = 100
class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        x_range = torch.zeros(2, 2)
        x_range[:,1] = 2.
        self.kan1 = KANLayer(2, neuron, base_num = base_num, x_range = x_range, train_knot = False)
        self.kan2 = KANLayer(neuron, neuron, base_num = base_num)
        self.kan3 = KANLayer(neuron, neuron, base_num = base_num)
        self.kan4 = KANLayer(neuron, 1, base_num = base_num, batch_norm = False)

    def forward(self, x):
        h = self.kan1(x)
        h = self.kan2(h)
        h = self.kan3(h)
        y = self.kan4(h)

        return y
    
    def adjust(self, x):
        x1 = self.kan1(x) #(batch_num, in_features)
        self.kan2.adjust(x1)
        x2 = self.kan2(x1)
        self.kan3.adjust(x2)

net = KAN().to("cuda")
num = sum(p.numel() for p in net.parameters()) #パラメータ数

opt = torch.optim.Adam(net.parameters(), lr = 1e-2)
criterion = nn.MSELoss()

epochs = 1000
test_loss_history = []
train_loss_history = []
for e in range(epochs):
    net.train()
    pred = net(train_x)
    train_loss = criterion(pred, train_y)
    opt.zero_grad(); train_loss.backward(); opt.step()
    train_loss_history.append(train_loss.item())

    net.eval()
    test_pred = net(test_x)
    test_loss = criterion(test_pred, test_y)
    test_loss_history.append(test_loss.item())
    print(train_loss.item(), test_loss.item())
    
#     print(loss.item())
#     # if ((e % 20) == 0) and (e < 50):
#     #     net.kan1.add_bases(net.kan1.base_num*2)
#         # net.adjust(x)

print(np.min([train_loss_history]))
print(np.min([test_loss_history]))
loss_history = np.stack((np.array(train_loss_history), np.array(test_loss_history)), axis = 1)
loss_history = pd.DataFrame(loss_history)
loss_history.to_csv("loss.csv")
print(num)