"""
discription:
    画像分類のサンプルコード
dataset:
    CIFAR-10
"""
from dataset import get_dataloader
from model import Net
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

train_data = get_dataloader()
test_data = get_dataloader(train = False)

net = Net().to("cuda")
opt = optim.Adam(net.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()


log_loss_train = []
log_loss_test = []
epochs = 50
for epoch in range(epochs):
    loss_sum = 0.; cnt = 0
    net.train()
    for x, y in train_data:
        pred = net(x.to("cuda"))
        loss = criterion(pred, y.to("cuda"))
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item(); cnt += 1
    log_loss_train.append(loss_sum / cnt)

    net.eval()
    loss_sum = 0.; cnt = 0
    net.train()
    for x, y in train_data:
        pred = net(x.to("cuda"))
        loss = criterion(pred, y.to("cuda"))
        loss_sum += loss.item(); cnt += 1
    log_loss_test.append(loss_sum / cnt)

    print("epoch:"+str(epoch))
    print(log_loss_train[-1])
    print(log_loss_test[-1])

torch.save(net.state_dict(), "./weight.pth")
data = np.stack([np.arange(epochs), np.array(log_loss_train), np.array(log_loss_test)], axis = -1)
data = pd.DataFrame(data)
data.to_csv("log_loss.csv")