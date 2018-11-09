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

class GarmetDataset(Dataset):
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

        # self.per_class_counts = [len(os.listdir(cf)) for cf in self.class_filepaths]
        
        # if masked:
        #     self.maskfolder_filepaths = [os.path.join(self.root, c, 'mask') for c in self.classes]
        #     self.maskframe_filepaths = []
        #     for folder in self.maskfolder_filepaths:
        #         self.maskframe_filepaths += [os.path.join(folder, filename) for filename in os.listdir(folder)]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((240, 320)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0], [0]),
            # torchvision.transforms.Normalize((np.iinfo(np.int64).max()/2,), (np.iinfo(np.int64).max,)),
        ])

        # test cases
        # print(self.get_class_index(0))    # class 0
        # print(self.get_class_index(1563)) # class 0
        # print(self.get_class_index(1564)) # class 1
        # print(self.get_class_index(3185)) # class 1 (last index)
        # print(self.get_class_index(3186)) # error
        # print(self.get_class_index(3187)) # error

    def __len__(self):
        ssum = 0
        for cclass in self.classes:
            ssum += len(os.listdir(os.path.join(self.root, cclass)))
        return ssum

    # def get_class_index(self, query_index):
    #     class_index = -1
    #     running_count = 0
    #     for pcc in self.per_class_counts:
    #         class_index += 1
    #         running_count += pcc
    #         if query_index < running_count:
    #             return class_index

    #     # bad default
    #     return -1

    def __getitem__(self, index):       
        #
        # Old way of masking images in-place
        # now we use pre-masked images
        #
        # image = io.imread(self.depthframe_filepaths[index], dtype='uint8') / 256
        # image = np.asarray(image)
        # image = Image.open(self.depthframe_filepaths[index])
        # if self.masked:
        #     mask = io.imread(self.maskframe_filepaths[index], dtype='uint8') / 256
        #     image = image*mask

        image = io.imread(self.frame_filepaths[index], dtype='uint8')

        # transform from (H, W) to (H, W, C) for a friendly format for ToPILImage
        image = np.expand_dims(image, axis=2)

        image = self.transform(image)
        # label = self.get_class_index(index)
        label = self.labels[index]
        
        return (image, label)