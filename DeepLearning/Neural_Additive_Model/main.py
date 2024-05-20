from dataset import get_train_val
from model import NAM
import sys

train_loader, val_loader = get_train_val()
in_features = 8 #入力次元

net = NAM(in_features).to("cuda")
for x, y in train_loader:
    pred = net(x.to("cuda"))
    print(pred.shape)
    sys.exit()