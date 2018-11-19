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

from datasets.xtion1 import Xtion1Dataset

class KinectDataset(Xtion1Dataset):
    # def __init__(self, root, masked=False):
    #     print(masked)
    #     super(KinectDataset, self).__init__(root, masked)

    def __getitem__(self, index):
        image = io.imread(self.frame_filepaths[index], dtype='float32')
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
        label = self.labels[index]
        
        return (image, label)

# # calculate training mean and std for kinect
# for t, c in xtion1_train_loader:
#     torch.std(t, dim=1)
#     # print(t[0].numpy())
#     print(np.mean(t[0].numpy()))