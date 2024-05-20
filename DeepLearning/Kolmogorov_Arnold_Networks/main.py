from model import KANLayer
import torch
import torch.nn as nn
import math
import pandas as pd

class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        x_range = torch.zeros(1, 2)
        x_range[0,1] = 2.*math.pi
        self.kan1 = KANLayer(1, 5, x_range = x_range)
        self.kan2 = KANLayer(5, 5)
        self.kan3 = KANLayer(5, 1, batch_norm = False)

    def forward(self, x):
        h = self.kan1(x)
        h = self.kan2(h)
        y = self.kan3(h)

        return y
    
    def adjust(self, x):
        x1 = self.kan1(x) #(batch_num, in_features)
        self.kan2.adjust(x1)
        x2 = self.kan2(x1)
        self.kan3.adjust(x2)


net = KAN()#.to("cuda")
num = sum(p.numel() for p in net.parameters()) #パラメータ数

opt = torch.optim.Adam(net.parameters(), lr = 1e-3)
criterion = nn.MSELoss()

epochs = 1001
x = torch.linspace(0., 2.*math.pi, 50).view(-1, 1)
y = torch.sin(x)
for e in range(1, epochs):
    pred = net(x)
    loss = criterion(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(loss.item())
    if ((e % 20) == 0) and (e < 50):
        net.kan1.add_bases(net.kan1.base_num*2)
        # net.adjust(x)


test_x = torch.linspace(0., 2.*math.pi, 100).view(-1, 1)
test_y = torch.sin(test_x)
test = net(test_x).detach()
data = torch.cat((test_x, test_y, test), dim = 1).numpy()
data = pd.DataFrame(data)
data.to_csv("test.csv")