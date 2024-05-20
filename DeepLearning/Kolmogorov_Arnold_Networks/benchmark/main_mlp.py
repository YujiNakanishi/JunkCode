from dataset import *
from MLP import SIREN, ReLU, Snake, FINER
import torch
import pandas as pd

epochs = 1000
model = Snake().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr = 1e-2)
criterion = torch.nn.MSELoss()

loss_history = []
for e in range(epochs):
    model.train()
    opt.zero_grad()
    train_pred = model(train_x)
    train_loss = criterion(train_pred, train_y)
    train_loss.backward(); opt.step()
    # loss_history.append(loss.item())

    model.eval()
    test_pred = model(test_x)
    test_loss = criterion(test_pred, test_y)
    print(test_loss.item())


# loss_history = pd.Series(loss_history)
# loss_history.to_csv("loss.csv")
# num = sum(p.numel() for p in model.parameters()) #パラメータ数
# print(num)