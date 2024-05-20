import torch
import torch.nn as nn
import torch.nn.functional as F


class ExU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((in_features, out_features)))
        torch.nn.init.normal_(self.w, mean = 3.5, std = 0.5)
        self.b = torch.nn.Parameter(torch.empty(in_features))
        torch.nn.init.normal_(self.w, mean = 0., std = 0.5)
    
    def forward(self, x):
        exu = (x - self.b) @ torch.exp(self.w)
        return torch.clip(exu, 0., 1.)

class subNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.exu1 = ExU(1, 32)
        self.exu2 = ExU(32, 32)
        self.exu3 = ExU(32, 1)
    
    def forward(self, x):
        h = self.exu1(x)
        h = self.exu2(h)
        y = self.exu3(h)

        return y


class NAM(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.subnets = nn.ModuleList([subNet() for i in range(in_features)])
        self.b = torch.nn.Parameter(torch.zeros(1))

        self.dropout = torch.nn.Dropout(p=0.5)

        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        F = torch.cat([self.subnets[i](x[:,i].view(-1, 1)) for i in range(self.in_features)], dim = 1)
        F = self.dropout(F)
        y = torch.sum(F, dim = 1) + self.b

        return y.view(-1, 1)
    
    def get_loss(self, x):
        """
        正則化項を入れた方が良いらしい。
        """
        pass