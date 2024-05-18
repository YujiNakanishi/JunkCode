import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

"""
/*****************/
mish
/*****************/
type : class. mish activation function
"""
class mish(nn.Module):

	def forward(self, x):
		return x*torch.tanh(F.softplus(x))

class Snake(nn.Module):

	def __init__(self, alpha = 1., requiresGrad = False):
		super().__init__()
		self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
		self.alpha.requires_grad = requiresGrad

	def forward(self, x):
		return x+torch.pow(torch.sin(self.alpha*x), 2.)/self.alpha


class Sinusoidal(nn.Module):

	def __init__(self, alpha = 1., beta = 1., requiresGrad = False):
		super().__init__()
		self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
		self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

		self.alpha.requires_grad = requiresGrad
		self.beta.requires_grad = requiresGrad

	def forward(self, x):
		return self.alpha*torch.sin(self.beta*x)