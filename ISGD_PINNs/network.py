import torch
import torch.nn as nn
import config as c


class PINNs(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features = 2, out_features = 100)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(in_features = 100, out_features = 100)
        self.act2 = nn.Tanh()
        self.lin3 = nn.Linear(in_features = 100, out_features = 100)
        self.act3 = nn.Tanh()
        self.lin4 = nn.Linear(in_features = 100, out_features = 100)
        self.act4 = nn.Tanh()
        self.lin5 = nn.Linear(in_features = 100, out_features = 100)
        self.act5 = nn.Tanh()
        self.lin6 = nn.Linear(in_features = 100, out_features = 100)
        self.act6 = nn.Tanh()
        self.lin7 = nn.Linear(in_features = 100, out_features = 1)
    
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.act5(self.lin5(x))
        x = self.act6(self.lin6(x))

        return self.lin7(x)
    
    def BC_loss(self):
        _input1x = torch.zeros(c.BC_num, 1)
        _input1y = torch.rand(c.BC_num, 1)
        _input1 = torch.cat((_input1x, _input1y), dim = 1)

        _input2x = torch.ones(c.BC_num, 1)
        _input2y = torch.rand(c.BC_num, 1)
        _input2 = torch.cat((_input2x, _input2y), dim = 1)
        
        _input3x = torch.rand(c.BC_num, 1)
        _input3y = torch.zeros(c.BC_num, 1)
        _input3 = torch.cat((_input3x, _input3y), dim = 1)

        _input4x = torch.rand(c.BC_num, 1)
        _input4y = torch.ones(c.BC_num, 1)
        _input4 = torch.cat((_input4x, _input4y), dim = 1)

        _input = torch.cat((_input1, _input2, _input3, _input4)).to("cuda")
        u = self(_input)
        loss = torch.mean(u**2)

        return loss
    
    def PDE_loss(self):
        _input = torch.rand(c.PDE_num, 2).to("cuda")
        _input.requires_grad = True
        u = self(_input)

        gradu = torch.autograd.grad(torch.sum(u), _input, create_graph=True)[0]
        dxxu = torch.autograd.grad(torch.sum(gradu[:,0]), _input, create_graph=True)[0][:,0]
        dyyu = torch.autograd.grad(torch.sum(gradu[:,1]), _input, create_graph=True)[0][:,1]

        f = (17.*(torch.pi**2) + 16.)*torch.sin(torch.pi*_input[:,0])*torch.sin(4*torch.pi*_input[:,1])

        residual = dxxu + dyyu + 16.*u[:,0] - f
        loss = torch.mean(residual**2)

        return loss