from PIL import Image
import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None):

        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        nx = 128
        ny = 128
        file_load = np.fromfile(fn, dtype='u1')
        if file_load.shape[0] != nx * ny:
            file_load = np.reshape(file_load, (512, 512))
            file_load = cv2.resize(file_load, (128, 128))
        file_load = np.reshape(file_load, (nx, ny, 1))
        if self.transform is not None:
            img = self.transform(file_load)
        return img, label

    def __len__(self):
        return len(self.imgs)


# train_data = MyDataset('./train.txt', transform=transforms.ToTensor())
# batch_size = 4
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# imshow(torchvision.utils.make_grid(images))

