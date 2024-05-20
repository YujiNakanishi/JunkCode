import torch
import torch.nn as nn
from activation import ReLU_n

"""
ExU layer
ref : https://arxiv.org/pdf/2004.13912
"""
class ExU(nn.Module):
    def __init__(self, in_features, out_features, n = 1.):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.normal_(self.weight, mean = 4., std = 0.5)
        self.bias = nn.Parameter(torch.empty(in_features))
        nn.init.normal_(self.bias, mean = 0., std = 0.5)

        self.relu_n = ReLU_n(n)
    
    def forward(self, x):
        h = (x - self.bias) @ torch.exp(self.weight)
        return self.relu_n(h)

class WeightSum(nn.Module):
	def __init__(self, init_val = None, num = None):
		super().__init__()
		if init_val is None:
			self.num = num
			self.w = nn.Parameter(torch.ones(num))
		else:
			self.num = len(init_val)
			self.w = nn.Parameter(init_val)

	def forward(self, X):
		result = 0.
		for i in range(self.num):
			result += self.w[i]*X[i]

		return result