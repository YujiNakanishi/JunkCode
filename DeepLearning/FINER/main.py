from dataset import get_Dataset
from model import SIREN, ReLU, Snake, FINER, FINE_Snake, FINER_x
import torch
import pandas as pd
import sys

epochs = 10001
save_interval = 1000

x, y = get_Dataset()

model = FINER_x().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = torch.nn.MSELoss()

test_x = torch.linspace(-2.*torch.pi, 2.*torch.pi, 100, device = "cuda").view(-1, 1)
test_y = torch.sin(test_x) + 0.3*torch.sin(10.*test_x)
test_pred = [test_x, test_y]
for e in range(epochs):
    opt.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward(); opt.step()

    if (e % save_interval) == 0:
        test_pred.append(model(test_x).detach())

test_pred = torch.cat(test_pred, dim = 1).cpu().numpy()
test_pred = pd.DataFrame(test_pred)
test_pred.to_csv("test.csv")