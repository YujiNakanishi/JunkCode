import torch
import torchvision

import config as c

def get_dataloader(train = True, download = True):
    dataset = torchvision.datasets.MNIST(
        root = "./",
        train = train,
        transform = c.transform,
        download = True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = c.batch_size,
        shuffle = True
    )

    return dataloader


if __name__ == "__main__":
    train_data = get_dataloader()