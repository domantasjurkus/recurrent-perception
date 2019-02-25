import os
import PIL
import numpy as np
from scipy import misc
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
import torchvision

def append_move_filepath(move_filepath, video_frames):
    return list(map(lambda frame_filename: os.path.join(move_filepath, frame_filename), video_frames))

class VideoEvaluationDataset(Dataset):
    def __init__(self, root, frames_per_sequence):
        self.root = root
        self.classes = os.listdir(self.root)
        self.video_filepaths = []
        self.labels = []

        for c in self.classes:
            class_filepath = os.path.join(self.root, c)
            for move_number in os.listdir(class_filepath):
                move_filepath = os.path.join(class_filepath, move_number)
                frames = sorted(os.listdir(move_filepath))
                frames = append_move_filepath(move_filepath, frames)
                self.video_filepaths.append(frames)
                self.labels.append(self.classes.index(c))

        #         n_sequences = len(frames) // frames_per_sequence
        #         for i in range(0, n_sequences):
        #             sequence = frames[i*frames_per_sequence:(i+1)*frames_per_sequence]
        #             sequence = append_move_filepath(move_filepath, sequence)
        #             self.sequence_filepaths.append(sequence)
        #             self.labels.append(self.classes.index(c))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.video_filepaths)

    def filepath_to_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = image[None, :,:] # add channel dimension
        # image = self.transform(image)
        return image

    def __getitem__(self, index):
        video = self.video_filepaths[index]
        video = list(map(self.filepath_to_image, video))
        video = np.asarray(video)
        video = torch.tensor(video)
        label = self.labels[index]        
        return (video, label)
