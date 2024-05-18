import torch
import torch.nn as nn

class RMSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
	def forward(self, pred, ans):
		return torch.sqrt(self.mse(pred, ans))