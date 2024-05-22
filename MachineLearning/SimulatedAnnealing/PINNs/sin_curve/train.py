import numpy as np
import torch
import model
import copy
import pandas as pd


def sample_sin(num = 20):
    x = 2.*np.pi*np.random.rand(num)
    y = np.sin(x)

    return x.reshape((-1, 1)), y.reshape((-1, 1))

x, y = sample_sin()
x = torch.tensor(x, dtype = torch.float32).to("cuda")
y = torch.tensor(y, dtype = torch.float32).to("cuda")

def getLoss(x, y, net):
    pred = net(x)
    loss = torch.mean((y-pred)**2)
    return float(loss)

#####decide initial temperature
p = 0.8
net = model.Net_sin(10).to("cuda")
sigma = 1e-3
num = 0
dLoss = []
while num < 100:
    loss = getLoss(x, y, net)
    net_new = copy.deepcopy(net)
    net_new.step(sigma)
    loss_new = getLoss(x, y, net_new)

    if loss_new > loss:
        dLoss.append(loss_new - loss)
        num += 1
    net = net_new

loss_mean = np.mean(dLoss)
T = -loss_mean/np.log(p)




net = model.Net_sin(10).to("cuda")
loss = getLoss(x, y, net)
min_loss = loss
best_params = net.state_dict()


dLoss = []
for itr in range(10000):
    net_new = copy.deepcopy(net)
    net_new.step(sigma)
    loss_new = getLoss(x, y, net_new)

    if loss_new < loss:
        net = copy.deepcopy(net_new)
        loss = loss_new
        if loss < min_loss:
            best_params = net.state_dict()

    elif np.random.rand() < np.exp(-(loss_new - loss)/T):
        net = copy.deepcopy(net_new)
        loss = loss_new
    
    T = 0.95*T
    dLoss.append(loss)
    print(str(itr)+"\t"+str(loss))

dLoss = pd.Series(dLoss)
dLoss.to_csv("loss.csv")

torch.save(best_params, "best_params.pth")