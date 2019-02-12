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
import torchvision.transforms as transforms

from datasets.xtion1 import Xtion1Dataset
from datasets.kinect import KinectDataset

from models.simple import SimpleNetwork
from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased
from train_test_singleshot import *
from util import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("device:", device)

MASKED = True
SHUFFLE = True
BATCH_SIZE = 64
# TEST_SPLIT = 0.8

ROOT_TRAIN = '../project-data/singleshot_%s' % ("masked" if MASKED else "depth")
ROOT_TEST = '../project-data/singleshot_%s_test' % ("masked" if MASKED else "depth")

# classes = ('pant', 'shirt', 'sweater', 'tshirt')
classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)
print("n_classes =", n_classes)

dataset_train = Xtion1Dataset(root=ROOT_TRAIN, classes=classes)
dataset_test = Xtion1Dataset(root=ROOT_TEST, classes=classes)
# keepaway_dataset = Xtion1Dataset(root='../project-data/singleshot_depth_keepaway', classes=classes, masked=MASKED)

# train_sampler, test_sampler = get_train_test_samplers(dataset_train)
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=train_sampler)
train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE)

# test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=test_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=BATCH_SIZE)

# keepaway_loader = torch.utils.data.DataLoader(keepaway_dataset, batch_size=BATCH_SIZE)

def get_model():
    model = CifarBased(n_classes=n_classes)
    # model.load_state_dict(torch.load('saved_models/cifarbased_unmasked_epoch20.pt'))
    # model = SimpleNetwork(n_classes=n_classes)
    # model = ResnetBased(n_classes=n_classes)
    # model = UPCFeatureExtractor(n_classes=n_classes)
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, train_loader, test_loader, n_classes, epochs=50, masked=MASKED, save=True, device=device)

    # test(model, keepaway_loader, n_classes, 1, device=device)
