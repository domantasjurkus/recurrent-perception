import os
import PIL
import numpy as np
from scipy import misc
from skimage import io, transform
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision

# def loader(filepath):
#     with open(filepath, 'rb') as f:
#         img = PIL.Image.open(f)
#         return img.convert('L')

# def get_training_data_iterator():
#     folder = GarmetDataset(root='../project-data/single_folder/training', masked=MASKED)
#     return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

# def get_testing_data_iterator():
#     folder = GarmetDataset(root='../project-data/single_folder/testing', masked=MASKED)
#     return iter(DataLoader(folder, batch_size=BATCH_SIZE, num_workers=4, shuffle=True))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((240, 320)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0, 0, 0), (65535.0, 65535.0, 65535.0)),
])

class GarmetDataset(Dataset):
    def __init__(self, root, masked=False, duplicate_channel=False):
        self.root = root
        self.classes = os.listdir(self.root)
        self.masked = masked
        self.duplicate_channel = duplicate_channel
        
        self.depthfolder_filepaths = [os.path.join(self.root, c, 'depth') for c in self.classes]
        self.depthframe_filepaths = []
        for folder in self.depthfolder_filepaths:
            self.depthframe_filepaths += [os.path.join(folder, filename) for filename in os.listdir(folder)]
        
        self.per_class_counts = [len(os.listdir(a)) for a in self.depthfolder_filepaths]
        
        if masked:
            self.maskfolder_filepaths = [os.path.join(self.root, c, 'mask') for c in self.classes]
            self.maskframe_filepaths = []
            for folder in self.maskfolder_filepaths:
                self.maskframe_filepaths += [os.path.join(folder, filename) for filename in os.listdir(folder)]

        # test cases
        # print(self.get_class_index(0))    # class 0
        # print(self.get_class_index(1563)) # class 0
        # print(self.get_class_index(1564)) # class 1
        # print(self.get_class_index(3185)) # class 1 (last index)
        # print(self.get_class_index(3186)) # error
        # print(self.get_class_index(3187)) # error

    def __len__(self):
        ssum = 0
        for cclass in self.classes:
            ssum += len(os.listdir(os.path.join(self.root, cclass, 'depth')))
        return ssum

    def get_class_index(self, query_index):
        class_index = -1
        running_count = 0
        for pcc in self.per_class_counts:
            class_index += 1
            running_count += pcc
            if query_index < running_count:
                return class_index

        # bad default
        return -1

    def __getitem__(self, index):
        # images = [self.loader(os.path.join(root, 'folder_{}'.format(i), self.image_folders[i][index])) for i in range(1, 9)]
        # images = [self.loader(os.path.join(root, f, self.image_folders[f][index])) for f in ['pant', 'shirt']]
        # image = Image.open(self.depthframe_filepaths[index])
        
        # this is not sorted...
        image = io.imread(self.depthframe_filepaths[index], dtype='uint8')
        if self.masked:
            mask = io.imread(self.maskframe_filepaths[index], dtype='uint8')
            image = image*mask

        # works with grayscale
        image = np.expand_dims(image, axis=2)

        # if self.duplicate_channel:
        #     image = np.stack((image,)*3, axis=-1)

        # https://github.com/pytorch/vision/issues/48
        # image = np.transpose(image,(2,0,1))
        image = transform(image)
        label = self.get_class_index(index)
        
        return (image, label)