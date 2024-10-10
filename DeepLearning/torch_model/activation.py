import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

class Absolute(nn.Module):
    """
    x = torch.linspace(x_min, x_max, num)に対する出力と導関数値をDataFrameで返す。
    """
    def test(self, x_min, x_max, num):
        x = torch.linspace(x_min, x_max, num)
        x.requires_grad = True

        y = self(x); torch.sum(y).backward()

        data = torch.stack((x, y, x.grad), dim = 1).detach().numpy()
        data = pd.DataFrame(data, columns = ["x", "y", "dxy"])

        return data

"""
上限nのあるrelu
ref : https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf
"""
class ReLU_n(Absolute):
    def __init__(self, n = 1.):
        super().__init__()
        self.n = n
    
    def forward(self, x):
        return torch.clamp(x, 0., self.n)

"""
Snake関数
ref : https://arxiv.org/pdf/2006.08195
"""
class Snake(Absolute):
    def __init__(self, a):
        super().__init__()
        self.a = a
    
    def forward(self, x):
        return x - 0.5*(torch.cos(2.*self.a*x) - 1.)/self.a
    
"""
ref : https://arxiv.org/pdf/2003.09855
"""
class TanhExp(Absolute):
    def forward(self, x):
        return x*torch.tanh(torch.exp(x))

class FINER(Absolute):
    def forward(self, x):
        return torch.sin(x*(torch.abs(x) + 1.))