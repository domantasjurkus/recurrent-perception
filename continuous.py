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

from datasets.xtion1continuous import Xtion1ContinuousDataset
from models.cnn_lstm import CNNLSTMModel
from models.cifar_based import CifarBased
from continuous.train_test_continuous import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("device:", device)

BATCH_SIZE = 8
SHUFFLE = True
TEST_SPLIT = 0.8
FRAMES_PER_SEQUENCE = 6

# classes = ('pant', 'shirt', 'sweater', 'tshirt')
classes = ('pant', 'shirt')
n_classes = len(classes)

xtion1_dataset = Xtion1ContinuousDataset(frames_per_sequence=FRAMES_PER_SEQUENCE)

def are_train_and_test_indices_separated(indices1, indices2):
    "test case to make sure we're not mixing testing and training samples"
    s = set(indices1+indices2)
    return len(s) == len(indices1)+len(indices2)
    
def get_train_test_samplers(dataset):
    size = len(dataset)
    indices = list(range(size))
    if SHUFFLE:
        np.random.seed(1337)
        np.random.shuffle(indices)
    split = int(np.floor(TEST_SPLIT*size))

    train_indices, test_indices = indices[:split], indices[split:]
    # print("Train and test indices separated: ", are_train_and_test_indices_separated(train_indices, test_indices))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler

train_sampler, test_sampler = get_train_test_samplers(xtion1_dataset)

train_params = {
    "batch_size": BATCH_SIZE,
    "sampler": train_sampler,
    "num_workers": 8
}

test_params = {
    "batch_size": BATCH_SIZE,
    "sampler": test_sampler,
    "num_workers": 8
}

train_loader = torch.utils.data.DataLoader(xtion1_dataset, **train_params)
test_loader = torch.utils.data.DataLoader(xtion1_dataset, **test_params)

def get_model():
    feature_extractor = CifarBased()
    feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_masked_50epochs.pt'))
    model = CNNLSTMModel(feature_extractor, n_classes)
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, train_loader, test_loader, n_classes, epochs=10)
