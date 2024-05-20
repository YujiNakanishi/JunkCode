import torch
import network
import config as c
import pandas as pd
import copy
import sys

net = network.PINNs().to("cuda")
net_clone = copy.deepcopy(net)
net_clone.eval()
opt = torch.optim.Adam(net.parameters(), lr = c.lr)

loss_history = []

for n in range(c.K0):
    for k in range(c.K1):
        opt.zero_grad()

        pinns_loss = net.BC_loss() + net.PDE_loss()

        implicit_loss = 0.
        for param, param_clone in zip(net.parameters(), net_clone.parameters()):
            implicit_loss += 0.5*torch.sum((param - param_clone)**2)
        
        loss = implicit_loss + c.alpha*pinns_loss
        loss.backward()
        opt.step()

    loss_history.append(pinns_loss.item())
    print(pinns_loss.item())
    net_clone = copy.deepcopy(net)

for n in range(c.K0, c.K2 + c.K0):
    opt.zero_grad()
    loss = net.BC_loss() + net.PDE_loss()
    loss.backward()
    opt.step()

    loss_history.append(loss.item())
    print(loss.item())

loss_history = pd.Series(loss_history)
loss_history.to_csv("isgd_adam.csv")