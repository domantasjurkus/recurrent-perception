import os
import PIL
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision

def append_move_filepath(move_filepath, sequence_frames):
    return list(map(lambda frame_filename: os.path.join(move_filepath, frame_filename), sequence_frames))

class Xtion1SnippetDataset(Dataset):
    def __init__(self, root, frames_per_sequence):
        self.root = root
        self.classes = os.listdir(self.root)
        self.sequence_filepaths = []
        self.labels = []

        for c in self.classes:
            class_filepath = os.path.join(self.root, c)
            for move_number in os.listdir(class_filepath):
                move_filepath = os.path.join(class_filepath, move_number)
                frames = sorted(os.listdir(move_filepath))
                n_sequences = len(frames) // frames_per_sequence
                for i in range(0, n_sequences):
                    sequence = frames[i*frames_per_sequence:(i+1)*frames_per_sequence]
                    sequence = append_move_filepath(move_filepath, sequence)
                    self.sequence_filepaths.append(sequence)
                    self.labels.append(self.classes.index(c))

        print("Total number of sequences:", len(self.sequence_filepaths))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.sequence_filepaths)

    def filepath_to_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = image[None, :,:] # add channel dimension
        # image = self.transform(image)
        return image

    def __getitem__(self, index):
        sequence = self.sequence_filepaths[index]
        sequence = list(map(self.filepath_to_image, sequence))
        sequence = np.asarray(sequence)
        sequence = torch.tensor(sequence)
        label = self.labels[index]   
        return (sequence, label)
