import os
import PIL
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import GarmetDataset
from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased
from models.simple import SimpleNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

WIDTH = 320
HEIGHT = 240
BATCH_SIZE = 64
MASKED = True

def show_image(tensor):
    print(tensor.shape)
    x = tensor.view(-1, HEIGHT, WIDTH)
    plt.imshow(x[0, ...])
    plt.colorbar()
    plt.show()

classes = ('pant', 'shirt')
n_classes = len(classes)

def get_training_data_iterator():
    folder = GarmetDataset(root='../project-data/single_folder/training', masked=MASKED, duplicate_channel=True)
    return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

def get_testing_data_iterator():
    dataset = GarmetDataset(root='../project-data/single_folder/testing', masked=MASKED, duplicate_channel=True)
    return iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

# cifar
# model = CifarBased(n_classes=n_classes)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# resnet
# model = ResnetBased(n_classes=n_classes)
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# simple
model = SimpleNetwork(n_classes=n_classes)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)
criterion = nn.CrossEntropyLoss()

def train():
    train_itr = get_training_data_iterator()
    print('training minibatch count:', len(train_itr))
    for epoch in range(1):  # loop multiple times if you want
        running_loss = 0.0
        for i, data in enumerate(train_itr):
            inputs, targets = data
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')
    # save_model()

def test():
    correct = 0
    total = 0
    # class_correct = list(0. for i in range(n_classes))
    # class_total = list(0. for i in range(n_classes))
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    test_itr = get_testing_data_iterator()
    print('testing minibatch count:', len(test_itr))

    with torch.no_grad():
        for i, data in enumerate(test_itr):
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            print(targets)

            outputs = model(images)

            _, predicted_indexes = torch.max(outputs.data, 1)
            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indexes == targets)
            correct += correct_vector.sum().item()
            
            for i in range(len(targets)):
                target = targets[i]
                class_correct[target] += correct_vector[i].item()
                class_total[target] += 1

    # Per class accuracy
    for i in range(n_classes):
        if class_total[i] != 0:
            print('Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i] / class_total[i]))
    
    print('Overall accuracy: %d %%' % (100 * correct / total))

def load_or_train():
    if os.path.exists('saved_models/model'):
        print('loading existing model')
        model.load_state_dict(torch.load('saved_models/model'))
    else:
        train()

def save_model():
    try:
        os.stat('saved_models')
    except:
        os.makedirs('saved_models')
    if not os.path.exists('saved_models/model'):
        torch.save(model.state_dict(), './saved_models/model')
        print("model saved")

# train()
# load_or_train()
# test()

training_itr = get_training_data_iterator()
data, labels = training_itr.next()
show_image(data)