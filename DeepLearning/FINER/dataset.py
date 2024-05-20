import torch

def get_Dataset():
    x = torch.linspace(-2.*torch.pi, 2.*torch.pi, 100).view(-1, 1)
    y = torch.sin(x) + 0.3*torch.sin(10.*x)

    return x.to("cuda"), y.to("cuda")