import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(train = True, batch_size = 32, transform = None, download = True):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
    dataset = torchvision.datasets.MNIST(
        root = "./",
        train = train,
        transform = transform,
        download = True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True
    )

    return dataloader