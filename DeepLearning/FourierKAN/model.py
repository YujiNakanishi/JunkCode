import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_Dataset
import numpy as np
import math

import sys

class SIREN(nn.Module):
    def __init__(self, alpha = 1.):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha.requires_grad = True
    def forward(self, x):
        return torch.sin(self.alpha*x)

class SNAKE(SIREN):
    def forward(self, x):
        return x + (torch.sin(self.alpha*x)**2)/self.alpha

class FINER(nn.Module):
    def forward(self, x):
        return torch.sin(x*(torch.abs(x) + 1.))

class MLP(nn.Module):
    def __init__(self,
                layer_num = 4,
                latent_dim = 32,
                act = nn.ReLU(),
                ):
        super().__init__()
        self.layer_num = layer_num
        self.latent_dim = latent_dim
        self.act = act

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, latent_dim))
        for _ in range(layer_num - 1):
            self.layers.append(nn.Linear(latent_dim, latent_dim))
        self.layers.append(nn.Linear(latent_dim, 1))

        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        
        x = self.layers[-1](x)
        return x
    
    @property
    def num_params(self):
        params = 0
        for p in self.parameters():
            if p.requires_grad:
                params += p.numel()
        
        return params
    
    def train(self, train_x, train_y, test_x, test_y, epochs = 1000):
        log_loss_test = []
        for epoch in range(epochs):
            pred = self(train_x)
            loss_train = self.criterion(pred, train_y)
            self.opt.zero_grad(); loss_train.backward(); self.opt.step()

            pred = self(test_x)
            loss_test = self.criterion(pred, test_y)
            log_loss_test.append(loss_test.item())
        
        return np.array(log_loss_test)
    
    def prediction(self, x):
        pred = self(x)
        return pred[:,0].detach().cpu().numpy()


class fkan_layer(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                K = 8):
        super().__init__()
        self.K = K
        self.a_params = nn.ParameterList([nn.Parameter((2.*torch.rand(K) - 1.)/(K*out_features)) for _ in range(out_features*in_features)])
        self.b_params = nn.ParameterList([nn.Parameter((2.*torch.rand(K) - 1.)/(K*out_features)) for _ in range(out_features*in_features)])
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        x_sin = torch.stack(tuple(torch.sin(k*x) for k in range(self.K)), dim = -1)
        x_cos = torch.stack(tuple(torch.cos(k*x) for k in range(self.K)), dim = -1)

        y = []
        for o_f in range(self.out_features):
            yi = 0.
            for i_f in range(self.in_features):
                a = torch.diag(self.a_params[self.in_features*o_f + i_f])
                b = torch.diag(self.b_params[self.in_features*o_f + i_f])
                yi += x_sin[:,i_f]@a + x_cos[:,i_f]@b
            
            y.append(torch.sum(yi, axis = 1))
        
        y = torch.stack(y, dim = 1)

        return y


class FKAN(MLP):
    def __init__(self,
                layer_num = 4,
                latent_dim = 32):
        super().__init__()
        self.layer_num = layer_num
        self.latent_dim = latent_dim

        self.layers = nn.ModuleList()
        self.layers.append(fkan_layer(1, latent_dim))
        for _ in range(layer_num - 1):
            self.layers.append(fkan_layer(latent_dim, latent_dim))
        self.layers.append(fkan_layer(latent_dim, 1))

        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr = 1e-3)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x