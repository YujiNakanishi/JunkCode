import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = "same", padding_mode = "reflect")
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 3, padding = "same", padding_mode = "reflect")
        self.lin1 = nn.Linear(in_features = 16*16*16, out_features = 100)
        self.lin2 = nn.Linear(in_features = 100, out_features = 10)
    
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.conv2(h)
        h = h.view(-1, 16*16*16)
        h = F.relu(self.lin1(h))

        return self.lin2(h)