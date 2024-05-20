import torch
import torch.nn as nn
import torch.nn.functional as F

class Absolute(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 1)

        self.criterion = nn.MSELoss()
        self.x_l = torch.zeros(1, 1, device = "cuda")
        self.x_r = (2.*torch.pi)*torch.ones(1, 1, device = "cuda")
    
    def act(self, x):
        return x
    
    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.act(self.lin2(h))
        y = self.lin3(h)

        return y
    
    def get_loss(self):
        ####loss at x = 0
        loss_l = self(self.x_l)**2
        loss_r = (1. - self(self.x_r))**2

        x_rand = torch.rand(100, 1, device = "cuda")*2.*torch.pi
        x_rand.requires_grad = True

        phi = self(x_rand)
        dxphi = torch.autograd.grad(torch.sum(phi), x_rand, retain_graph=True, create_graph=True)[0]
        dxxphi = torch.autograd.grad(torch.sum(dxphi), x_rand, retain_graph=True, create_graph=True)[0]

        pde_loss = torch.mean((dxxphi - torch.sin(x_rand) - 10.*torch.sin(10.*x_rand))**2)

        return pde_loss + loss_l + loss_r


class SIREN(Absolute):
    def act(self, x):
        return torch.sin(2.*x)

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

class FINE_x(FINER):
    def act(self, x):
        return torch.sin(x*(torch.abs(x) + 1.)) + x

class FINE_sigmoid(Absolute):
    def act(self, x):
        h = F.sigmoid(x)
        return torch.sin(h*(h + 1.)) + x