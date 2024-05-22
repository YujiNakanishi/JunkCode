import numpy as np
from pyevtk.hl import imageToVTK
import torch

import model

net = model.SA_Net().to("cuda")
net.load_state_dict(torch.load("params_SA_pinns.pth"))

xi = np.arange(101)*0.01
x, y = np.meshgrid(xi, xi)
x = torch.tensor(x.reshape((-1, 1)), dtype = torch.float32)
y = torch.tensor(y.reshape((-1, 1)), dtype = torch.float32)
input = torch.cat((x, y), dim = 1).to("cuda")

u = net(input).detach().cpu().numpy().reshape((101,101, 1))

point_data = {"u" : u}
imageToVTK("./test_SA_pinns", pointData = point_data)