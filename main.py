import os
import PIL
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms, datasets

# get rid of these if possible
from dataloaders.dataloader import GarmetDataset
# from kinect_dataloader import KinectDataset

from models.cifar_based import CifarBased
from models.resnet_based import ResnetBased
from models.simple import SimpleNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

WIDTH = 320
HEIGHT = 240
BATCH_SIZE = 64
MASKED = True
SHUFFLE = True
TEST_SPLIT = 0.9

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

def intercept(img):
    array = torch.from_numpy(np.array(img, np.float64, copy=True))
    # print(array[100])
    return img

# transform = transforms.Compose([
#     transforms.Lambda(intercept),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# xtion1_dataset = datasets.ImageFolder(root='../project-data/xtion1/depth', transform=transforms.ToTensor())
# xtion1_dataset = datasets.ImageFolder(root='../project-data/xtion1/depth', transform=transform)
xtion1_dataset = GarmetDataset(root='../project-data/xtion1', masked=MASKED)

# split indices and create random test and train samplers
xtion1_size = len(xtion1_dataset)
xtion1_indices = list(range(xtion1_size))
if SHUFFLE:
    print("shuffle")
    np.random.seed(1337)
    np.random.shuffle(xtion1_indices)
split = int(np.floor(TEST_SPLIT*xtion1_size))

train_indices, test_indices = xtion1_indices[:split], xtion1_indices[split:]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

xtion1_train_loader = torch.utils.data.DataLoader(xtion1_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
xtion1_test_loader = torch.utils.data.DataLoader(xtion1_dataset, batch_size=BATCH_SIZE)

# # calculate training mean and std
# for t, c in xtion1_train_loader:
#     torch.std(t, dim=1)
#     # print(t[0].numpy())
#     print(np.mean(t[0].numpy()))

# image_means = [np.mean(t[0].numpy()) for t, _ in xtion1_train_loader]
# print(image_means)
# print(np.mean(image_means))

# 
# display a sample
#
test_image_batch, labels = iter(xtion1_test_loader).next()
test_image = test_image_batch[0][0]
show_image(test_image)

# kinect_dataset = KinectDataset(root='../project-data/kinect', masked=MASKED)
# kinect_loader = torch.utils.data.DataLoader(kinect_dataset, batch_size=BATCH_SIZE)

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
def train(train_loader, test_loader, epochs=10):
    # train_itr = train_loader
    minibatch_count = len(train_loader)
    print('training minibatch count:', minibatch_count)
    for epoch in range(epochs):

        total_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 5 == 0:
                print('epoch: %d minibatches: %d/%d loss: %.3f' % (epoch+1, i+1, minibatch_count, total_loss))
                # running_loss = 0.0
        
        training_losses.append(total_loss)
        test(test_loader)

    print('Finished Training')
    # save_model()

def test(test_loader):
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    total_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    minibatch_count = len(test_loader)
    print('testing minibatch count:', minibatch_count)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            # bin predictions into confusion matrix
            for j in range(len(images)):
                actual = targets[j].item()
                predicted = predicted_indexes[j].item()
                confusion[actual][predicted] += 1

            # sum up total correct
            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indexes == targets)
            correct += correct_vector.sum().item()
            
            # sum up per-class correct
            for j in range(len(targets)):
                target = targets[j]
                class_correct[target] += correct_vector[j].item()
                class_total[target] += 1

            if i % 5 == 0:
                print('minibatches: %d/%d loss: %.3f' % (i+1, minibatch_count, total_loss))
        
        testing_losses.append(total_loss)

    print('Correct predictions:', class_correct)
    print('Total test samples: ', class_total)
    print('Test accuracy: %d %%' % (100 * correct / total))
    print(confusion)

train(xtion1_train_loader, xtion1_test_loader, 10)
print('Training losses:', training_losses)
print('Testing losses:', testing_losses)
# load_or_train()
# test(xtion1_test_loader)







#
# attempts to normalise histograms
#
def filter_out(img2d):
    limit = 1.0/128
    return list(filter(lambda pix: pix > limit, img2d))

# train_itr = iter(train_loader)
# train_batch = train_itr.next()
# train_image = train_batch[0][0][0].detach().numpy()
# kinect_image = iter(kinect_loader).next()[0][0][0] .detach().numpy()
# kinect_image = kinect_image.astype('uint8')
# print(train_image[100])

# This thing is giving me a headache, will use numpy implementation
# cv2.equalizeHist(kinect_image)

# clahe = cv2.createCLAHE()
# cl1 = clahe.apply(kinect_image)
# show_image(train_image)

# from util import normalise_histogram
# normalise_histogram(train_image)
# plt.show()

# train_filtered = filter_out(train_image.ravel())
# kinect_filtered = filter_out(kinect_image.ravel())
# print(len(kinect_filtered))

# plt.hist(train_image.ravel(), bins=256, range=(0.0, 1.0))
# plt.show()
# plt.hist(kinect_image.ravel(), bins=128, range=(0.0, 1.0))
# plt.show()
