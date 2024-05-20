"""
discription:
    画像データに対する拡散モデル
input:
    MNISTデータセット
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
import pandas as pd
import numpy as np
from model import Unet, Diffuser

import sys

train_loader = get_dataloader()
net = Unet().to("cuda")
opt = optim.Adam(net.parameters(), lr = 1e-4)
diffuser = Diffuser()
log_loss = []
criterion = nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    loss_sum = 0.; cnt = 0
    for x, _ in train_loader:
        x = x.to("cuda")
        y = net.low_resolution(x)
        
        ts = torch.randint(1, diffuser.num_timestep + 1, (len(x), ), device = "cuda")

        xt, noise = diffuser.add_noise(x, ts)
        pred_noise = net(xt, ts, y)
        loss = criterion(pred_noise, noise)

        opt.zero_grad(); loss.backward(); opt.step()

        loss_sum += loss.item(); cnt += 1
        print(str(epoch)+"\t"+str(loss.item()))
    
    log_loss.append(loss_sum / cnt)

torch.save(net.state_dict(), "./weight.pth")
data = np.stack([np.arange(epochs), np.array(log_loss)], axis = -1)
data = pd.DataFrame(data)
data.to_csv("log_loss.csv")