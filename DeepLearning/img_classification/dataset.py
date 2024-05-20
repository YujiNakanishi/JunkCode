import torch
import torchvision
import torchvision.transforms as transforms

"""
DataLoaderの作成
input:
    batch_size -> <int> ミニバッチ数
    transform -> 
    train -> <bool> 訓練データか評価用データか
    download -> <bool> データをダウンロードするか否か
"""
def get_dataloader(batch_size = 32, transform = None, train = True, download = True):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])

    dataset = torchvision.datasets.CIFAR10(
        root = "./",
        train = train,
        transform = transform,
        download = True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
    )

    return dataloader