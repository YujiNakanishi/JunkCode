"""
pinns学習で比較
"""
from model import SIREN, Snake, FINER, FINE_x, FINE_sigmoid
import torch
import pandas as pd

epochs = 5001
save_interval = 1000

model = SIREN().to("cuda")
num = sum(p.numel() for p in model.parameters()) #パラメータ数
# print(num)
opt = torch.optim.Adam(model.parameters(), lr = 1e-5)

test_x = torch.linspace(0., 2.*torch.pi, 100, device = "cuda").view(-1, 1)
test_y = -torch.sin(test_x) - 0.1*torch.sin(10.*test_x) + test_x/2./torch.pi
test_pred = [test_x, test_y]
for e in range(epochs):
    loss = model.get_loss()
    opt.zero_grad(); loss.backward(); opt.step()
    print(loss.item())

    if (e % save_interval) == 0:
        test_pred.append(model(test_x).detach())

test_pred = torch.cat(test_pred, dim = 1).cpu().numpy()
test_pred = pd.DataFrame(test_pred)
test_pred.to_csv("test.csv")