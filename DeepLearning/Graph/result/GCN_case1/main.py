from module import dataset, Networks
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyvista as pv

train_list, test_list = dataset.train_test_split()
train_dataset = dataset.Coord(train_list)
test_dataset = dataset.Coord(test_list)
train_loader = DataLoader(train_dataset, batch_size = 8)
test_loader = DataLoader(test_dataset, batch_size = 8)

epochs = 1000
net = Networks.GCN(32, 32, 3, 5).to("cuda")
criterion = nn.MSELoss()
opt = optim.Adam(net.parameters(), lr = 1e-3)

trainloss_history = []
testloss_history = []
best_state = None
best_loss = float("inf")


for epoch in range(epochs):
    loss_sum = 0.
    count = 0    
    for train_data in train_loader:
        preds = net(train_data)
        loss = criterion(preds, train_data.y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
        count += 1
    trainloss_history.append(loss_sum/count)

    with torch.no_grad():
        loss_sum = 0.
        count = 0
        for test_data in test_loader:
            preds = net(test_data)
            loss = criterion(preds, test_data.y)
            loss_sum += loss.item()
            count += 1
        testloss_history.append(loss_sum/count)

        if testloss_history[-1] < best_loss:
            torch.save(net.state_dict(), "weight.pth")
            best_loss = testloss_history[-1]
            
    print("epoch = {0} : train_loss = {1}, test_loss = {2}".format(epoch, trainloss_history[-1], testloss_history[-1]))

trainloss_history = np.array(trainloss_history)
testloss_history = np.array(testloss_history)
loss_history = np.stack((
    np.arange(epochs),
    trainloss_history,
    testloss_history
), axis = 1)

loss_history = pd.DataFrame(loss_history)
loss_history.to_csv("loss_history.csv")


net.load_state_dict(torch.load("weight.pth"))
data = test_dataset.get(0)
pred = net(data).detach().cpu().numpy()
y = data.y.cpu().numpy()
x_interp = data.x.cpu().numpy()


filename = test_dataset.data_list[0]["file_dir"]+"/{}mm.vtk".format(test_dataset.data_list[0]["HR_scale"])
geo = pv.UnstructuredGrid(pv.read(filename))
geo.points = y
geo.save("ans.vtk") #正解結果
geo.points = pred
geo.save("pred.vtk") #推論結果
geo.points = x_interp
geo.save("interp.vtk") #単に補間したときの結果。

diff_interp = np.mean(np.linalg.norm(y-x_interp, axis = 1)) #補間結果と正解結果のずれ
diff = np.mean(np.linalg.norm(y-pred, axis = 1)) #推論結果と正解結果のずれ
print(diff_interp)
print(diff)