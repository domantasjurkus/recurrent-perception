import os
import PIL
import numpy as np
from scipy import misc
import skimage
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision

class Xtion1Dataset(Dataset):
    def __init__(self, root, masked=False):
        self.root = os.path.join(root, 'masked' if masked else 'depth')
        self.classes = os.listdir(self.root)
        self.frame_filepaths = []
        self.labels = []

        for c in self.classes:
            class_filepath = os.path.join(self.root, c)
            for filename in os.listdir(class_filepath):
                self.frame_filepaths.append(os.path.join(class_filepath, filename))
                self.labels.append(self.classes.index(c))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # resize otherwise run out of memory
            torchvision.transforms.Resize((240, 320)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0], [0]),
            # torchvision.transforms.Normalize((np.iinfo(np.int64).max()/2,), (np.iinfo(np.int64).max,)),
        ])

    def __len__(self):
        ssum = 0
        for cclass in self.classes:
            ssum += len(os.listdir(os.path.join(self.root, cclass)))
        return ssum

    def __getitem__(self, index):       
        image = io.imread(self.frame_filepaths[index], dtype='uint8')

        # transform from (H, W) to (H, W, 1) for a friendly format for ToPILImage
        image = np.expand_dims(image, axis=2)

        image = self.transform(image)
        label = self.labels[index]
        
        return (image, label)