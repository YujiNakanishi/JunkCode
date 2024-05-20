from KAN import KANLayer
import torch
import torch.nn as nn
import math
import pandas as pd


"""
KANによるPINNs。
PINNsのときはbatch norm使わない方がいいかも。B.C.ロス計算するとき、入力xのミニバッチデータはすごく偏る。
"""
class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        x_range = torch.zeros(1, 2)
        x_range[0,1] = 2.*math.pi
        self.kan1 = KANLayer(1, 10, base_num = 10, x_range = x_range, train_knot = False)
        self.kan2 = KANLayer(10, 10, base_num = 10)
        self.kan3 = KANLayer(10, 10, base_num = 10)
        self.kan4 = KANLayer(10, 1, base_num = 10, batch_norm = False)

        self.kan5 = KANLayer(1, 10, base_num = 100, x_range = x_range, train_knot = False)
        self.kan6 = KANLayer(10, 10, base_num = 100)
        self.kan7 = KANLayer(10, 10, base_num = 100)
        self.kan8 = KANLayer(10, 1, base_num = 100, batch_norm = False)

        self.criterion = nn.MSELoss()
        self.x_l = torch.zeros(1, 1, device = "cuda")
        self.x_r = (2.*torch.pi)*torch.ones(1, 1, device = "cuda")
    
    def forward(self, x):
        h1 = torch.tanh(self.kan1(x))
        h1 = torch.tanh(self.kan2(h1))
        h1 = torch.tanh(self.kan3(h1))
        y1 = self.kan4(h1)

        h2 = torch.tanh(self.kan5(x))
        h2 = torch.tanh(self.kan6(h2))
        h2 = torch.tanh(self.kan7(h2))
        y2 = self.kan8(h2)

        return y1 + y2
    
    def get_loss(self):
        loss_l = self(self.x_l)**2
        loss_r = (1. - self(self.x_r))**2

        x_rand = torch.rand(100, 1, device = "cuda")*2.*torch.pi
        x_rand.requires_grad = True

        phi = self(x_rand)
        dxphi = torch.autograd.grad(torch.sum(phi), x_rand, retain_graph=True, create_graph=True)[0]
        dxxphi = torch.autograd.grad(torch.sum(dxphi), x_rand, retain_graph=True, create_graph=True)[0]

        pde_loss = torch.mean((dxxphi - torch.sin(x_rand) - 10.*torch.sin(10.*x_rand))**2)

        return pde_loss + loss_l + loss_r


epochs = 5001
save_interval = 1000
model = KAN().to("cuda")
num = sum(p.numel() for p in model.parameters()) #パラメータ数
opt = torch.optim.Adam(model.parameters(), lr = 1e-5)

test_x = torch.linspace(0., 2.*torch.pi, 100, device = "cuda").view(-1, 1)
test_y = -torch.sin(test_x) - 0.1*torch.sin(10.*test_x) + test_x/2./torch.pi
test_pred = [test_x, test_y]
for e in range(epochs):
    loss = model.get_loss()
    opt.zero_grad(); loss.backward(); opt.step()
    print(e, loss.item())

    if (e % save_interval) == 0:
        test_pred.append(model(test_x).detach())

test_pred = torch.cat(test_pred, dim = 1).cpu().numpy()
test_pred = pd.DataFrame(test_pred)
test_pred.to_csv("test.csv")