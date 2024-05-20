import PIL
import torch
import numpy as np

class Imageset(torch.utils.data.Dataset):
	def __init__(self, img_files, labels = None, transform = None):
		self.img_files = img_files
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, idx):
		img = PIL.Image.open(self.img_files[idx])

		if not(self.transform is None):
			img = self.transform(img)

		if self.labels is None:
			label = torch.clone(img)
		else:
			label = self.labels[idx]

		return img, label

class NPYset(torch.utils.data.Dataset):
	def __init__(self, npy_files, labels = None, transpose = None):
		self.npy_files = npy_files
		self.labels = labels
		self.transpose = transpose

	def __len__(self):
		return len(self.npy_files)

	def transform(self, npy):
		return npy

	def __getitem__(self, idx):
		npy = np.load(self.npy_files[idx])
		npy = npy.astype("float32")
		if not(self.transpose is None):
			npy = npy.transpose(self.transpose)

		npy = self.transform(npy)
		npy = torch.tensor(npy)

		if self.labels is None:
			label = torch.clone(npy)
		else:
			label = self.labels[idx]

		return npy, label