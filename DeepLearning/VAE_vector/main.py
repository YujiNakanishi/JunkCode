"""
discription:
    数ベクトルデータに対するVAEコード
input:
    MNISTデータセット
"""

import torch
import torch.optim as optim
from dataset import *
from model import VAE
import pandas as pd
import numpy as np

train_loader = get_dataloader() #訓練データローダ
test_loader = get_dataloader(train = False) #テストデータローダ

net = VAE().to("cuda")
opt = optim.Adam(net.parameters(), lr = c.lr)
log_loss_train = []
log_loss_test = []

for epoch in range(c.epochs):
    loss_sum = 0.; cnt = 0
    net.train()
    for x, _ in train_loader:
        loss = net.get_loss(x.to("cuda"))
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); cnt += 1
    log_loss_train.append(loss_sum / cnt)

    net.eval()
    loss_sum = 0.; cnt = 0
    for x, _ in test_loader:
        loss = net.get_loss(x.to("cuda"))
        loss_sum += loss.item(); cnt += 1
    log_loss_test.append(loss_sum / cnt)

    print("epoch:"+str(epoch))
    print(log_loss_train[-1])
    print(log_loss_test[-1])

torch.save(net.state_dict(), c.weight_path)
data = np.stack([np.arange(c.epochs), np.array(log_loss_train), np.array(log_loss_test)], axis = -1)
data = pd.DataFrame(data)
data.to_csv("log_loss.csv")