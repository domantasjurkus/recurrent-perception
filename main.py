import os
import PIL
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets.xtion1 import Xtion1Dataset
from datasets.kinect import KinectDataset

from models.simple import SimpleNetwork
from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased

from train_test import *
from util import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

MASKED = True

BATCH_SIZE = 64
SHUFFLE = True
TEST_SPLIT = 0.8

classes = ('pant', 'shirt', 'sweater', 'tshirt')
n_classes = len(classes)
print("n_classes =", n_classes)

# xtion1_dataset = torchvision.datasets.ImageFolder(root='../project-data/xtion1/depth', transform=torchvision.transforms.ToTensor())
# xtion1_dataset = torchvision.datasets.ImageFolder(root='../project-data/xtion1/depth', transform=transform)
xtion1_dataset = Xtion1Dataset(root='../project-data/xtion1', masked=MASKED)

def get_train_test_samplers(dataset):
    size = len(dataset)
    indices = list(range(size))
    if SHUFFLE:
        print("shuffle")
        np.random.seed(1337)
        np.random.shuffle(indices)
    split = int(np.floor(TEST_SPLIT*size))

    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler

# train_sampler, test_sampler = get_train_test_samplers(xtion1_dataset)
# xtion1_train_loader = torch.utils.data.DataLoader(xtion1_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# xtion1_test_loader = torch.utils.data.DataLoader(xtion1_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

# 
# Only for the mixed approach suggested by Gerardo
# 
xtion1_dataset_mixed = Xtion1Dataset(root='../project-data/xtion1_mixed', masked=True)
xtion1_dataset_test = Xtion1Dataset(root='../project-data/xtion1', masked=False)
xtion1_train_loader = torch.utils.data.DataLoader(xtion1_dataset_mixed, batch_size=BATCH_SIZE)
xtion1_test_loader = torch.utils.data.DataLoader(xtion1_dataset_test, batch_size=BATCH_SIZE)

# display a sample
# test_image, labels = iter(xtion1_test_loader).next()[0][0]
# show_image(test_image)

def get_model():
    model = CifarBased(n_classes=n_classes)
    # model = SimpleNetwork(n_classes=n_classes)
    # model = ResnetBased(n_classes=n_classes)
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, xtion1_train_loader, xtion1_test_loader, n_classes, epochs=30, masked=MASKED)

    # k_dataset = KinectDataset(root='../project-data/kinect_masked_subset', masked=MASKED)
    # k_loader = torch.utils.data.DataLoader(k_dataset, batch_size=4)
