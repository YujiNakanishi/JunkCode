"""
Download MNIST dataset as png files.

*Directory structure
./MNIST
    -train/
        - 0/
        - 1/
        - 2/
        ...
    -test/

The label of any images are same with name of folder.
"""

import os
from torchvision import datasets

root_dir = "./MNIST/"
train_dir = root_dir + "train/"
test_dir = root_dir + "test/"
os.makedirs(root_dir)
os.makedirs(train_dir)
os.makedirs(test_dir)


for i in range(10):
    os.makedirs(train_dir + str(i)+"/")
    os.makedirs(test_dir + str(i)+"/")



train_data = datasets.MNIST(root = root_dir, train = True, transform = None, download = True)
test_data = datasets.MNIST(root = root_dir, train = False, transform = None, download = True)

num_img = 0
for img, label in train_data:
    img.save(train_dir+str(label)+"/img_"+str(num_img)+".png")
    num_img += 1

num_img = 0
for img, label in test_data:
    img.save(test_dir+str(label)+"/img_"+str(label)+".png")
    num_img += 1