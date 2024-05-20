import pandas as pd
import numpy as np
import torch
import sys

import torch.utils

def transform(x):
    """
    変数の定義域を[-1,1]に変換
    Input : x -> <np:float:(N, )>
    """
    x_min = np.min(x); x_max = np.max(x)
    x_new = 2.*(x - x_min)/(x_max - x_min) - 1.

    return x_new

def get_train_val():
    data = pd.read_csv("data.csv").values
    X = data[:,:-1]; y = data[:,-1]

    y = transform(y).reshape((-1, 1))
    X = np.stack([transform(X[:,i]) for i in range(X.shape[1])], axis = 1)

    X = torch.tensor(X, dtype = torch.float, device = "cuda"); y = torch.tensor(y, dtype = torch.float, device = "cuda")
    dataset = torch.utils.data.TensorDataset(X, y)
    ratio = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ratio, 1. - ratio])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle = True)

    return train_loader, val_loader