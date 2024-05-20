import torch
import torch.nn as nn
import torch.nn.functional as F

class Absolute(nn.Module):
    def __init__(self, k = None):
        super().__init__()
        self.lin1 = nn.Linear(2, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 32)
        self.lin4 = nn.Linear(32, 1)
    
    def act(self, x):
        return x
    
    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.act(self.lin2(h))
        h = self.act(self.lin3(h))
        y = self.lin4(h)

        return y


class SIREN(Absolute):
    def act(self, x):
        return torch.sin(x)

class ReLU(Absolute):
    def act(self, x):
        return F.relu(x)

class Snake(Absolute):
    def act(self, x):
        return x - 0.5*(torch.cos(4.*x) - 1.)/2. #a = 2

class FINER(Absolute):
    def __init__(self, k = 1.):
        super().__init__()
        nn.init.uniform_(self.lin1.bias, -k, k)
        nn.init.uniform_(self.lin2.bias, -k, k)
    
    def act(self, x):
        return torch.sin(x*(torch.abs(x) + 1.))

class FINER_x(FINER):
    def act(self, x):
        return torch.sin(x*(torch.abs(x) + 1.)) + x


class FINE_Snake(FINER):
    def act(self, x):
        a = 1.
        xx = x*(torch.abs(x) + 1.)
        return xx - 0.5*(torch.cos(2.*a*xx) - 1.)/a