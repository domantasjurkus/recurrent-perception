import os
import PIL
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import GarmetDataset
from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased
from models.simple import SimpleNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

WIDTH = 320
HEIGHT = 240
BATCH_SIZE = 128
MASKED = True
TEST_SPLIT = 0.2

def show_image(tensor):
    # print(tensor.shape)
    # x = tensor.view(-1, HEIGHT, WIDTH)
    x = tensor.detach().numpy()
    plt.imshow(x[...])
    # plt.colorbar()
    plt.show()

# def show_grid(minibatch):
#     grid = vutils.make_grid(minibatch, nrow=6)
#     transposed = grid.permute(1, 2, 0)
#     plt.imshow(transposed)
#     plt.colorbar()
#     plt.show()

# classes = ('pant', 'shirt')
classes = ('pant', 'shirt', 'sweater', 'tshirt')
n_classes = len(classes)
print("n_classes =", n_classes)

# def get_training_data_iterator():
#     folder = GarmetDataset(root='../project-data/single_folder/training', masked=MASKED)
#     return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

# def get_testing_data_iterator():
#     folder = GarmetDataset(root='../project-data/single_folder/testing', masked=MASKED)
#     return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

def get_dataset_iterator():
    folder = GarmetDataset(root='../project-data/single_folder', masked=MASKED)
    return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

dataset = GarmetDataset(root='../project-data/single_folder', masked=MASKED)
dataset_size = len(dataset)
indices = list(range(dataset_size))

SHUFFLE = True
if SHUFFLE :
    np.random.seed(219)
    np.random.shuffle(indices)

split = int(np.floor(TEST_SPLIT*dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

# cifar
model = CifarBased(n_classes=n_classes)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

# resnet
# model = ResnetBased(n_classes=n_classes)
# optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# simple
# model = SimpleNetwork(n_classes=n_classes)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)
criterion = nn.CrossEntropyLoss()

def train():
    train_itr = train_loader
    print('training minibatch count:', len(train_itr))
    for epoch in range(1):
        print("epoch", epoch)
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
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')
    # save_model()

def test():
    correct = 0
    total = 0
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    test_itr = test_loader
    
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    with torch.no_grad():
        for i, data in enumerate(test_itr):
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            print(len(images))
            for i in range(len(images)):
                actual = targets[i].item()
                predicted = predicted_indexes[i].item()
                print(actual, predicted)
                confusion[actual][predicted] += 1

            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indexes == targets)
            correct += correct_vector.sum().item()
            # print("corrects:", correct, '/', batch_size)
            
            for i in range(len(targets)):
                target = targets[i]
                class_correct[target] += correct_vector[i].item()
                class_total[target] += 1

    print(class_correct)
    print(class_total)    
    print('Overall accuracy: %d %%' % (100 * correct / total))
    print(confusion)

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

train()
# load_or_train()
test()

# def show_topmost_resnet_params(model):
#     k = list(model.parameters())[0]
#     g = vutils.make_grid(k, padding=5)

#     transposed = g.permute(1, 2, 0)
#     transposed = transposed.data
    
#     plt.imshow(transposed)
#     plt.colorbar()
#     plt.show()

# show_topmost_resnet_params(model)

# training_itr = get_training_data_iterator()
# images, labels = training_itr.next()
