import torch
import torch.nn as nn


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