import torch
import torch.nn as nn




class Net(nn.Module):
    def __init__(self, neuron = 100):
        super().__init__()

        self.lin1 = nn.Linear(in_features = 2, out_features = neuron)
        nn.init.xavier_normal_(self.lin1.weight)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.xavier_normal_(self.lin2.weight)
        self.act2 = nn.Tanh()
        self.lin3 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.xavier_normal_(self.lin3.weight)
        self.act3 = nn.Tanh()
        self.lin4 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.xavier_normal_(self.lin4.weight)
        self.act4 = nn.Tanh()
        self.lin5 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.xavier_normal_(self.lin5.weight)
        self.act5 = nn.Tanh()
        self.lin6 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.xavier_normal_(self.lin6.weight)
        self.act6 = nn.Tanh()
        self.lin7 = nn.Linear(in_features = neuron, out_features = 1)
        nn.init.xavier_normal_(self.lin7.weight)
    
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.act5(self.lin5(x))
        x = self.act6(self.lin6(x))
        
        return self.lin7(x)
    
    def BC_loss(self, num = 100):
        x1 = torch.zeros(num, 1)
        y1 = torch.rand(num, 1)
        input1 = torch.cat((x1, y1), dim = 1)

        x2 = torch.ones(num, 1)
        y2 = torch.rand(num, 1)
        input2 = torch.cat((x2, y2), dim = 1)

        x3 = torch.rand(num, 1)
        y3 = torch.zeros(num, 1)
        input3 = torch.cat((x3, y3), dim = 1)
        
        x4 = torch.rand(num, 1)
        y4 = torch.ones(num, 1)
        input4 = torch.cat((x4, y4), dim = 1)

        input = torch.cat((input1, input2, input3, input4)).to("cuda")

        u = self(input)

        loss = torch.mean(u**2)
        return loss
    
    def PDE_loss(self, num = 4000):
        input = torch.rand(num, 2).to("cuda")
        input.requires_grad = True

        u = self(input)
        gradu = torch.autograd.grad(torch.sum(u), input, create_graph=True)[0]
        dxxu = torch.autograd.grad(torch.sum(gradu[:,0]), input, create_graph=True)[0][:,0]
        dyyu = torch.autograd.grad(torch.sum(gradu[:,1]), input, create_graph=True)[0][:,1]

        f = (17.*torch.pi**2 + 16.)*torch.sin(torch.pi*input[:,0])*torch.sin(4.*torch.pi*input[:,1])
        residual = dxxu + dyyu + 16.*u - f

        return torch.mean(residual**2)


class SA_Net(Net):
    def __init__(self, neuron = 100):
        super().__init__(neuron)
        for param in self.parameters():
            param.requires_grad = False
    
    def step(self, sigma = 1.):
        for param in self.parameters():
            param += sigma*torch.randn(param.shape).to("cuda")