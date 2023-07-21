from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
from imageProcessing import DataAugmentation as DataAug
from torchvision import transforms
from torchvision.transforms import functional as F

class MyDataset(Dataset):
    def __init__(self, x, y=None, data_aug=False, aug_factor=0.5, img_size=(224, 224)):
        self.x = x
        self.y = y
        self.data_aug = data_aug
        self.aug_factor = aug_factor
        self.mean = [0.63013001, 0.4266026 , 0.53309676]
        self.std = [0.23022491, 0.26698795, 0.24416544]
        self.img_size = img_size

        if y is not None:
            assert x.shape[0] == y.shape[0], "Labels shape should be same with Images set"

        if self.data_aug:
            self.x = np.concatenate([self.x, self.x], axis=0)
            if y is not None:
                self.y = np.concatenate([self.y, self.y], axis=0)

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    size=self.img_size,
                    scale=(0.8, 1.0)
                    ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.2
                    ),
                transforms.RandomRotation(0.8),
                transforms.Resize(self.img_size),
            ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        if isinstance(self.x[index], str):
            img = cv2.imread(self.x[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img / 255 - self.mean) / self.std
        else:
            img = self.x[index].copy()

        # Expand 1 channel to 3 channels
        if img.shape[-1] == 1:
            img = np.concatenate((img, img, img), 2)

        # Data Augmentation
        if self.data_aug and (index > len(self) / 2):
            img_tensor = self.transform(img)
        else:
            img = cv2.resize(img, self.img_size)
            img_tensor = torch.tensor(img)
            img_tensor = img_tensor.permute(2, 0, 1) # [w, h, c] -> [c, w, h]
            
        if self.y is None: # Test mode
            return img_tensor
        else:
            label_tensor = torch.tensor(self.y[index], dtype=torch.int64)
            return img_tensor, label_tensor

# Unit test
if __name__=="__main__":
    img_root = "NtutEMnist/emnist-byclass-train.npz"
    
    data = np.load(img_root)
    train_labels = data['training_labels']
    train_images = data['training_images']

    trn_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    mean = np.mean(trn_images, axis=(1,2), keepdims=True)
    std = np.std(trn_images, axis=(1,2), keepdims=True)
    trn_images = (trn_images - mean) / std

    val_size = int(train_images.shape[0] * 0.1)
    x_val = trn_images[:val_size]
    y_val = train_labels[:val_size]
    x_train = trn_images[val_size:]
    y_train = train_labels[val_size:]

    dataset = MyDataset(x_train, y_train)
    for i, (img, label) in enumerate(dataset):
        print(img.shape, label.shape)
        break
