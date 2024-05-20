import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

def _pos_encoding(t, out_dim):
    v = torch.zeros(out_dim, device = "cuda")
    i = torch.arange(0, out_dim, device = "cuda")
    a = 10000**(i/out_dim)

    v[0::2] = torch.sin(t/a[0::2])
    v[1::2] = torch.cos(t/a[1::2])

    return v

def pos_encoding(ts, out_dim):
    """
    positional encoding
    input:
        ts -> <array:float> 時刻バッチデータ
        out_dim -> <int> 出力次元数
    output:
        v -> <torch:float:(len(ts), out_dim)>
    """
    v = torch.stack([_pos_encoding(t, out_dim) for t in ts], dim = 0)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pos_encoding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = "same")
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = "same")
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.lin1 = nn.Linear(pos_encoding_dim, in_channels)
        self.lin2 = nn.Linear(in_channels, in_channels)
    
    def forward(self, x, v):
        N, C, _, _ = x.shape

        v = F.relu(self.lin1(v))
        v = self.lin2(v)
        v = v.view(N, C, 1, 1)

        h = x + v
        h = self.conv1(h)
        h = self.batchnorm1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.batchnorm2(h)
        y = F.relu(h)

        return y


class Unet(nn.Module):
    def __init__(self, in_channels = 1, pos_encoding_dim = 100, num_labels = 10):
        super().__init__()
        self.pos_encoding_dim = pos_encoding_dim
        self.num_labels = num_labels
        self.down1 = ConvBlock(in_channels, 64, self.pos_encoding_dim)
        self.down2 = ConvBlock(64, 128, self.pos_encoding_dim)
        self.bottom = ConvBlock(128, 256, self.pos_encoding_dim)
        self.up1 = ConvBlock(384, 128, self.pos_encoding_dim)
        self.up2 = ConvBlock(192, 64, self.pos_encoding_dim)
        self.conv = nn.Conv2d(64, in_channels, 1)
        self.embedding = nn.Embedding(self.num_labels, self.pos_encoding_dim)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

    def forward(self, x, ts, labels):
        v = pos_encoding(ts, self.pos_encoding_dim)
        v += self.embedding(labels)

        h1 = self.down1(x, v)

        h2 = self.pool(h1)
        h2 = self.down2(h2, v)

        h3 = self.pool(h2)
        h3 = self.bottom(h3, v)

        h2_1 = self.upsample(h3)
        h2 = torch.cat((h2, h2_1), dim = 1)
        h2 = self.up1(h2, v)

        h1_1 = self.upsample(h2)
        h1 = torch.cat((h1, h1_1), dim = 1)
        h1 = self.up2(h1, v)

        x_out = self.conv(h1)

        return x_out


class Diffuser:
    def __init__(self, num_timestep = 1000, beta_start = 0.0001, beta_end = 0.02):
        self.num_timestep = num_timestep
        self.betas = torch.linspace(beta_start, beta_end, num_timestep, device = "cuda")
        self.alphas = 1. - self.betas
        self.alpha_cumprods = torch.cumprod(self.alphas, dim = 0)
    
    def add_noise(self, x0, ts):
        alpha_cumprod = self.alpha_cumprods[ts - 1]
        alpha_cumprod = alpha_cumprod.view(alpha_cumprod.size(0), 1, 1, 1)

        noise = torch.randn_like(x0, device = "cuda")
        xt = torch.sqrt(alpha_cumprod)*x0 + torch.sqrt(1. - alpha_cumprod)*noise

        return xt, noise
    
    def denoise(self, model, x, ts, labels):
        alpha = self.alphas[ts - 1]
        alpha_cumprod = self.alpha_cumprods[ts - 1]
        alpha_cumprod_prev = self.alpha_cumprods[ts - 2]

        alpha = alpha.view(alpha.size(0), 1, 1, 1)
        alpha_cumprod = alpha_cumprod.view(alpha.size(0), 1, 1, 1)
        alpha_cumprod_prev = alpha_cumprod_prev.view(alpha.size(0), 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, ts, labels)
        model.train()

        noise = torch.randn_like(x, device = "cuda")
        noise[ts == 1] = 0

        mu = (x - ((1 - alpha)/torch.sqrt(1 - alpha_cumprod))*eps) / torch.sqrt(alpha)
        std = torch.sqrt((1. - alpha)*(1. - alpha_cumprod_prev)/(1. - alpha_cumprod))
        return mu + noise*std
    
    def to_img(self, x):
        x *= 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        pil = transforms.ToPILImage()
        return pil(x)
    

    def sampling(self, model, return_img = False):
        x_shape = (64, 1, 28, 28)
        labels = np.concatenate([np.array([i]*8) for i in range(8)])
        labels = torch.tensor(labels, dtype = torch.long, device = "cuda")
        batch_size = x_shape[0]

        x = torch.randn(x_shape, device = "cuda")

        for i in range(self.num_timestep, 0, -1):
            t = i*torch.ones(batch_size, device = "cuda", dtype = torch.long)
            x = self.denoise(model, x, t, labels)
        if return_img:
            imgs = [self.to_img(xx) for xx in x]
            return imgs
        else:
            return x