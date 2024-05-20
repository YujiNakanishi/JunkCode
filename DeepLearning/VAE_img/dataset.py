"""
Create dataset by ImageFolder

Relationship between labels(int) and folder names can be seen by find_classes
>>>print(train_data.find_classes("./MNIST/train")) -> Tuple(list of class name, relationships)
(["0", "1", "2", "3", "4", ...], {"0":0, "1":1, "2":2, "3":3, "4":4, ...})
"""
import torchvision
import torchvision.transforms as transforms
import torch


def get_dataloader(train = True, transform = None, batch_size = 32):
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    
    dataset = torchvision.datasets.ImageFolder("./MNIST/train", transform = transform) if train else torchvision.datasets.ImageFolder("./MNIST/test", transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

    return dataloader