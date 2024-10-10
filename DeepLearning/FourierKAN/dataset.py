import torch
import numpy as np
import sys

# def get_Dataset(num = 1000, train_ratio = 0.8):
#     x = np.linspace(-2.*np.pi, 2.*np.pi, num)
#     y = np.sin(x) + 0.3*np.sin(10.*x) + 0.1*np.sin(30.*x) + 0.05*np.sin(50.*x)
#     train_num = int(num*train_ratio)
#     indice = np.arange(num)
#     np.random.shuffle(indice)

#     train_idx = indice[:train_num]; test_idx =indice[train_num:]
#     train_x = x[train_idx]; train_y = y[train_idx]
#     test_x = x[test_idx]; test_y = y[test_idx]

#     train_x = torch.tensor(train_x, dtype = torch.float, device = "cuda").view(-1, 1)
#     train_y = torch.tensor(train_y, dtype = torch.float, device = "cuda").view(-1, 1)
#     test_x = torch.tensor(test_x, dtype = torch.float, device = "cuda").view(-1, 1)
#     test_y = torch.tensor(test_y, dtype = torch.float, device = "cuda").view(-1, 1)

#     return train_x, train_y, test_x, test_y

def get_Dataset(num = 1000, train_ratio = 0.8, shuffle = True):
    x1 = np.linspace(-2.*np.pi, 0, num)[:-1]
    y1 = np.sin(x1)
    x2 = np.linspace(0., 2.*np.pi, num)
    y2 = np.sin(x2) + 0.3*np.sin(10.*x2) + 0.1*np.sin(30.*x2) + 0.05*np.sin(50.*x2)
    
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    train_num = int(len(x)*train_ratio)
    indice = np.arange(len(x))
    if shuffle:
        np.random.shuffle(indice)
    train_idx = indice[:train_num]; test_idx =indice[train_num:]
    train_x = x[train_idx]; train_y = y[train_idx]
    test_x = x[test_idx]; test_y = y[test_idx]

    train_x = torch.tensor(train_x, dtype = torch.float, device = "cuda").view(-1, 1)
    train_y = torch.tensor(train_y, dtype = torch.float, device = "cuda").view(-1, 1)
    test_x = torch.tensor(test_x, dtype = torch.float, device = "cuda").view(-1, 1)
    test_y = torch.tensor(test_y, dtype = torch.float, device = "cuda").view(-1, 1)

    return train_x, train_y, test_x, test_y