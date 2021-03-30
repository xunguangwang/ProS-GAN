import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms


class HashingDataset(Dataset):
    def __init__(self,
                 data_path,
                 img_filename,
                 label_filename,
                 transform=transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
                 ])):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()
