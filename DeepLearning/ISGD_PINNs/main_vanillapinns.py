import torch
import network
import config as c
import pandas as pd

net = network.PINNs().to("cuda")

opt = torch.optim.Adam(net.parameters(), lr = c.lr)

loss_history = []
for epoch in range(c.epochs):
    opt.zero_grad()

    bc_loss = net.BC_loss()
    pde_loss = net.PDE_loss()

    loss = bc_loss + pde_loss
    loss.backward()
    opt.step()

    loss_history.append(loss.item())
    print(str(epoch)+"\t"+str(loss.item()))

loss_history = pd.Series(loss_history)
loss_history.to_csv("vanilla_pinns.csv")