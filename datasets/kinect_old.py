import os
import PIL
import numpy as np
from scipy import misc
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision
from dataloader import GarmetDataset

class KinectDataset(GarmetDataset):
    def __getitem__(self, index):
        image = io.imread(self.depthframe_filepaths[index], dtype='float32')
        # self.masked = False
        # if self.masked:
        #     mask = io.imread(self.maskframe_filepaths[index], dtype='uint8') // 2
        #     image = image*mask
            
        # works with grayscale
        image = np.expand_dims(image, axis=2)

        # https://github.com/pytorch/vision/issues/48
        # image = np.transpose(image,(2,0,1))
        image = self.transform(image)
        # print(image.dtype)
        label = self.get_class_index(index)
        
        return (image, label)

kinect_dataset = KinectDataset(root='../project-data/kinect_parsed', masked=MASKED)
kinect_loader = torch.utils.data.DataLoader(kinect_dataset, batch_size=64)

img, label = itr(kinect_loader).next()

# # calculate training mean and std for kinect
# for t, c in xtion1_train_loader:
#     torch.std(t, dim=1)
#     # print(t[0].numpy())
#     print(np.mean(t[0].numpy()))