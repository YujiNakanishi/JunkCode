import torch
import torch.nn as nn
import torch.nn.functional as F
import config as c
import sys

class Encoder(nn.Module):
    def __init__(self,
        ):
        super().__init__()

        self.lin = nn.Linear(in_features = c.input_dim, out_features = c.hidden_dim)
        self.lin_mu = nn.Linear(in_features = c.hidden_dim, out_features = c.latent_dim) #期待値ベクトル計算のための層
        self.lin_logvar = nn.Linear(in_features = c.hidden_dim, out_features = c.latent_dim) #分散のlog値ベクトル計算のための層
    
    def forward(self, x):
        h = F.relu(self.lin(x))

        mu = self.lin_mu(h)
        logvar = self.lin_logvar(h)
        sigma = torch.exp(0.5*logvar) #torch.exp(logvar) = sigma^2なため
        
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self,
        ):
        super().__init__()
        self.lin1 = nn.Linear(in_features = c.latent_dim, out_features = c.hidden_dim)
        self.lin2 = nn.Linear(in_features = c.hidden_dim, out_features = c.input_dim)
    
    def forward(self, z):
        h = F.relu(self.lin1(z))
        x = F.sigmoid(self.lin2(h)) #各ピクセル値x_ijはx_ij \in [0, 1]なため、最後にsigmoidを通す。
        return x


def sampling(mu, sigma):
    e = torch.randn_like(sigma)
    return mu + e*sigma

class VAE(nn.Module):
    def __init__(self,
        ):
        super().__init__()
        self.encoder = Encoder(); self.decoder = Decoder()
    
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = sampling(mu, sigma)
        x_out = self.decoder(z)

        N = len(x) #バッチサイズ

        loss_mse = F.mse_loss(x_out, x, reduction = "sum")
        loss_klD = -torch.sum(1. + torch.log(sigma**2) - mu**2 - sigma**2) #KL-ダイバージェンス
        return (loss_mse + loss_klD) / N