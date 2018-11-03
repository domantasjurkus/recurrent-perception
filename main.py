import os
import PIL
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import cv2

from dataloader import GarmetDataset
from kinect_dataloader import KinectDataset
from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased
from models.simple import SimpleNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

WIDTH = 320
HEIGHT = 240
BATCH_SIZE = 128
MASKED = True
TEST_SPLIT = 0.3

def show_tensor(tensor, layer=0):
    # print(tensor.shape)
    # x = tensor.view(-1, HEIGHT, WIDTH)
    x = tensor.detach().numpy()
    plt.imshow(x[layer, ...])
    plt.colorbar()
    plt.show()

def show_image(img):
    plt.imshow(img)
    plt.colorbar()
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

def get_dataset_iterator():
    folder = GarmetDataset(root='../project-data/single_folder', masked=MASKED)
    return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

dataset = GarmetDataset(root='../project-data/single_folder', masked=MASKED)
dataset_size = len(dataset)
indices = list(range(dataset_size))

SHUFFLE = True
if SHUFFLE :
    np.random.seed(1337)
    np.random.shuffle(indices)

split = int(np.floor(TEST_SPLIT*dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

kinect_dataset = KinectDataset(root='../project-data/kinect', masked=MASKED)
kinect_loader = torch.utils.data.DataLoader(kinect_dataset, batch_size=BATCH_SIZE)

# cifar
# model = CifarBased(n_classes=n_classes)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()

# simple
model = SimpleNetwork(n_classes=n_classes)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()

model.to(device)

training_losses = []
testing_losses = []

torch.set_printoptions(precision=10)
def train(epochs=10):
    train_itr = train_loader
    print('training minibatch count:', len(train_itr))
    for epoch in range(epochs):
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
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss))
                # running_loss = 0.0
        
        training_losses.append(running_loss)
        test()

    print('Finished Training')
    # save_model()

def test(loader=test_loader):
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    running_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    print('testing...')
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            for i in range(len(images)):
                actual = targets[i].item()
                predicted = predicted_indexes[i].item()
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
        
        testing_losses.append(running_loss)

    print(class_correct)
    print(class_total)    
    print('Test accuracy: %d %%' % (100 * correct / total))
    print(confusion)

# train(3)
# print(training_losses)
# print(testing_losses)
# load_or_train()
# test(kinect_loader)
# test()    

def filter_out(img2d):
    return list(filter(lambda pix: pix > limit, img2d))

train_itr = iter(train_loader)
train_batch = train_itr.next()
train_image = train_batch[0][0][0].detach().numpy()
kinect_image = iter(kinect_loader).next()[0][0][0].detach().numpy()
kinect_image = kinect_image.astype('uint8')
print(train_image[100])

# This thing is giving me a headache, will use numpy implementation
# cv2.equalizeHist(kinect_image)

# clahe = cv2.createCLAHE()
# cl1 = clahe.apply(kinect_image)
# show_image(train_image)

# from util import normalise_histogram
# normalise_histogram(train_image)
# plt.show()

# limit = 1.0/128
# train_filtered = filter_out(train_image.ravel())
# kinect_filtered = filter_out(kinect_image.ravel())
# print(len(kinect_filtered))

# plt.hist(train_image.ravel(), bins=256, range=(0.0, 1.0))
# plt.show()
# plt.hist(kinect_image.ravel(), bins=128, range=(0.0, 1.0))
# plt.show()
