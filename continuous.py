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

# from datasets.xtion1continuous import Xtion1ContinuousDataset
from datasets.xtion1video import Xtion1VideoDataset
from models.cifar_based import CifarBased
# from models.lstm_snippet import LSTMSnippet
from models.lstm_video import LSTMVideo
from train_test_continuous import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

BATCH_SIZE = 1
SHUFFLE = False
TEST_SPLIT = 0.8
FRAMES_PER_SEQUENCE = 12

# classes = ('pant', 'shirt')
classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

# dataset_train = Xtion1ContinuousDataset(root='../project-data/continuous_depth', frames_per_sequence=FRAMES_PER_SEQUENCE)
# dataset_test = Xtion1ContinuousDataset(root='../project-data/continuous_depth_test', frames_per_sequence=FRAMES_PER_SEQUENCE)
# dataset_keepaway = Xtion1ContinuousDataset(root='../project-data/continuous_depth_keepaway', frames_per_sequence=FRAMES_PER_SEQUENCE)

dataset_train = Xtion1VideoDataset(root='../project-data/continuous_masked')
dataset_test = Xtion1VideoDataset(root='../project-data/continuous_masked_test')

train_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": True,
}

test_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": True,
}

loader_train = torch.utils.data.DataLoader(dataset_train, **train_params)
loader_test = torch.utils.data.DataLoader(dataset_test, **test_params)
# loader_keepaway = torch.utils.data.DataLoader(dataset_keepaway, batch_size=BATCH_SIZE, num_workers=8)

def get_model():
    feature_extractor = CifarBased(n_classes=n_classes)
    feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_masked_epoch11.pt'))

    model = CNNLSTMModel(feature_extractor, n_classes)
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=25, save=False, device=device)

    # test(model, loader_keepaway, n_classes, 1, device=device)
