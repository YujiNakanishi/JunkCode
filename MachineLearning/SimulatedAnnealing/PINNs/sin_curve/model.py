import torch
import torch.nn as nn

class Net_sin(nn.Module):
    def __init__(self, neuron = 10):
        super().__init__()
        self.lin1 = nn.Linear(in_features = 1, out_features = neuron)
        nn.init.uniform_(self.lin1.weight, -1., 1.); nn.init.uniform_(self.lin1.bias, -1., 1.)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.uniform_(self.lin2.weight, -1., 1.); nn.init.uniform_(self.lin2.bias, -1., 1.)
        self.act2 = nn.Tanh()
        self.lin3 = nn.Linear(in_features = neuron, out_features = neuron)
        nn.init.uniform_(self.lin3.weight, -1., 1.); nn.init.uniform_(self.lin3.bias, -1., 1.)
        self.act3 = nn.Tanh()
        self.lin4 = nn.Linear(in_features = neuron, out_features = 1)
        nn.init.uniform_(self.lin4.weight, -1., 1.); nn.init.uniform_(self.lin4.bias, -1., 1.)

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))

        return self.lin4(x)
    
    def step(self, sigma = 1.):
        for param in self.parameters():
            param += sigma*torch.randn(param.shape).to("cuda")