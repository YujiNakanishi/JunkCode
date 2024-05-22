import pandas as pd
import torch
import copy
import numpy as np

import model

sigma = 1e-5

"""
decide initial temperature
"""
p = 0.8
num = 0
dLoss = []

net = model.SA_Net().to("cuda")

while num < 100:
    loss = float(net.BC_loss() + net.PDE_loss())
    net_new = copy.deepcopy(net)
    net_new.step(sigma)
    loss_new = float(net_new.BC_loss() + net_new.PDE_loss())

    if loss_new > loss:
        dLoss.append(loss_new - loss)
        num += 1
    net = net_new

loss_mean = np.mean(dLoss)
T = -loss_mean/np.log(p)


"""
train
"""
net = model.SA_Net().to("cuda")
loss = float(net.BC_loss() + net.PDE_loss())
min_loss = loss
best_params = net.state_dict()

loss_history = []
for itr in range(10000):
    net_new = copy.deepcopy(net)
    net_new.step(sigma)
    loss_new = float(net_new.BC_loss() + net_new.PDE_loss())

    if loss_new < loss:
        net = copy.deepcopy(net_new)
        loss = loss_new
        if loss < min_loss:
            best_params = net.state_dict()
    
    elif np.random.rand() < np.exp(-(loss_new - loss)/T):
        net = copy.deepcopy(net_new)
        loss = loss_new
    loss_history.append(loss)
    print(str(itr)+"\t"+str(loss))
    
    T = 0.95*T

loss_history = pd.Series(loss_history)
loss_history.to_csv("loss_SA_pinns.csv")

torch.save(best_params, "params_SA_pinns.pth")