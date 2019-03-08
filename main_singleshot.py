import os
import torch
import torchvision
import torchvision.utils as vutils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets.xtion1 import Xtion1Dataset
from models.cifar_based import CifarBased
from train_test import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("device:", device)

MASKED = True
SHUFFLE = True
BATCH_SIZE = 64

# ROOT_TRAIN = '../project-data/singleshot_%s_resized' % ("masked" if MASKED else "depth")
# ROOT_TEST = '../project-data/singleshot_%s_test_resized' % ("masked" if MASKED else "depth")
ROOT_TRAIN = '../project-data/singleshot_%s' % ("masked" if MASKED else "depth")
ROOT_TEST = '../project-data/singleshot_%s_test' % ("masked" if MASKED else "depth")


classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)
print("n_classes =", n_classes)

dataset_train = Xtion1Dataset(root=ROOT_TRAIN, classes=classes)
dataset_test = Xtion1Dataset(root=ROOT_TEST, classes=classes)

train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=BATCH_SIZE)

def get_model():
    model = CifarBased(n_classes=n_classes)
    # model.load_state_dict(torch.load('saved_models/cifarbased_unmasked_epoch20.pt'))
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, train_loader, test_loader, n_classes, epochs=50, masked=MASKED, save=False, device=device)

    # test(model, keepaway_loader, n_classes, 1, device=device)
