import torch
import math

def data(x):
    return torch.exp(torch.sin(math.pi*x[:,0])+x[:,1]**2).view(-1, 1)


train_x = torch.rand(1000, 2, device = "cuda")
test_x = torch.rand(100, 2, device = "cuda")
train_y = data(train_x)
test_y = data(test_x)