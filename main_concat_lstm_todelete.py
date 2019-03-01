import torch
import random

# random.seed(1337)
# torch.manual_seed(1337)

# Unused I think

from datasets.xtion1video import Xtion1VideoDataset
from models.cifar_based import CifarBased
from models.lstm_video import LSTMVideo
from train_test_continuous import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

BATCH_SIZE = 1
SHUFFLE = True

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

dataset_train = Xtion1VideoDataset(root='../project-data/continuous_masked_resized')
dataset_test = Xtion1VideoDataset(root='../project-data/continuous_masked_test_resized')

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

def get_model():
    feature_extractor = CifarBased(n_classes=n_classes)
    feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_nodrop_masked_epoch6_acc35.941499.pt'))

    model = LSTMVideo(feature_extractor.features, n_classes)
    
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=100, save=True, masked=True, device=device)
