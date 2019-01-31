import os
import PIL
import numpy as np
from scipy import misc
import skimage
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
import torchvision

class Xtion1Dataset(Dataset):
    def __init__(self, root, classes, masked=False):
        # self.root = os.path.join(root, 'masked' if masked else 'depth')
        self.root = root
        # self.classes = os.listdir(self.root)
        self.classes = classes
        self.frame_filepaths = []
        self.labels = []
        print("Xtion1Dataset classes:", self.classes)

        for c in self.classes:
            class_filepath = os.path.join(self.root, c)
            for filename in os.listdir(class_filepath):
                self.frame_filepaths.append(os.path.join(class_filepath, filename))
                self.labels.append(self.classes.index(c))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # we need to do the normalisation below if our data has variable brightness
            # (comes from different sources)
            # torchvision.transforms.Normalize((np.iinfo(np.int64).max()/2,), (np.iinfo(np.int64).max,)),
        ])

    def __len__(self):
        ssum = 0
        for cclass in self.classes:
            ssum += len(os.listdir(os.path.join(self.root, cclass)))
        return ssum

    def __getitem__(self, index):
        # image = io.imread(self.frame_filepaths[index], dtype='uint8')
        image = cv2.imread(self.frame_filepaths[index], cv2.IMREAD_GRAYSCALE)

        image = image[:,:,None]
        image = self.transform(image)
        label = self.labels[index]
        
        return (image, label)