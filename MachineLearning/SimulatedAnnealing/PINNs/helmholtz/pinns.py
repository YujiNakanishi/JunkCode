import pandas as pd
import torch
import torch.optim as optim

import model

net = model.Net().to("cuda")
optimizer = optim.Adam(net.parameters(), lr = 1e-5)

loss_history = []
for itr in range(10000):
    optimizer.zero_grad()
    bc_loss = net.BC_loss()
    pde_loss = net.PDE_loss()
    loss = bc_loss + pde_loss

    loss.backward(); optimizer.step()

    print(str(itr) + "\t" + str(float(loss)))
    loss_history.append(float(loss))

loss_history = pd.Series(loss_history)
loss_history.to_csv("loss_pinns.csv")

torch.save(net.state_dict(), "params_pinns.pth")