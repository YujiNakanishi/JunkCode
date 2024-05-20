import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = "same", padding_mode = "reflect")
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = "same", padding_mode = "reflect")
        self.pool2 = nn.MaxPool2d(2)

        self.lin_mu = nn.Linear(in_features = 49*64, out_features = 256)
        self.lin_logvar = nn.Linear(in_features = 49*64, out_features = 256)

    
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool1(h)
        h = F.relu(self.conv2(h))
        h = self.pool2(h)

        h = h.view((-1, 49*64))

        mu = self.lin_mu(h)
        logvar = self.lin_logvar(h)
        sigma = torch.exp(0.5*logvar)

        return mu, sigma

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features = 256, out_features = 49*64)
        self.up1 = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = "same", padding_mode = "reflect")
        self.up2 = nn.Upsample(scale_factor = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, padding = "same", padding_mode = "reflect")

    def forward(self, z):
        h = F.relu(self.lin1(z))
        h = h.view((-1, 64, 7, 7))
        h = self.up1(h)
        h = F.relu(self.conv1(h))
        h = self.up2(h)

        return F.sigmoid(self.conv2(h))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(); self.decoder = Decoder()
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = mu + sigma*torch.randn_like(sigma)

        x_out = self.decoder(z)

        loss_mse = F.mse_loss(x_out, x, reduction = "sum")
        loss_klD = -torch.sum(1. + torch.log(sigma**2) - mu**2 - sigma**2)

        return (loss_mse + loss_klD) / len(x)