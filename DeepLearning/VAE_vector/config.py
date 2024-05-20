import torch
import torchvision.transforms as transforms

input_dim = 784 #入力次元。MNISTの場合784
hidden_dim = 200 #中間層次元数
latent_dim = 20 #潜在空間の次元数
batch_size = 32
lr = 3e-4
epochs = 30

weight_path = "./weight.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten),
    ])
