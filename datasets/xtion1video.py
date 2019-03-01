import os
import PIL
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torchvision

def append_move_filepath(move_filepath, frame_filepaths):
    return list(map(lambda frame_filename: os.path.join(move_filepath, frame_filename), frame_filepaths))

class Xtion1VideoDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = os.listdir(self.root)
        self.video_filepaths = []
        self.labels = []
        self.longest_video = 1

        for c in self.classes:
            class_filepath = os.path.join(self.root, c)
            for move_number in os.listdir(class_filepath):
                move_filepath = os.path.join(class_filepath, move_number)
                frame_filepaths = sorted(os.listdir(move_filepath))
                if len(frame_filepaths) > self.longest_video:
                    self.longest_video = len(frame_filepaths)
                video = append_move_filepath(move_filepath, frame_filepaths)
                self.video_filepaths.append(video)
                self.labels.append(self.classes.index(c))

        print("Total number of videos:", len(self.video_filepaths))

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
        frames = list(map(self.filepath_to_image, video))
        frames = np.asarray(frames)

        # padd with 0s
        # sorry for the hardcode (⌐■_■)
        video = torch.zeros((self.longest_video, 1, 480, 640))
        video[:len(frames)] = torch.Tensor(frames)
        video = torch.Tensor(video)

        # Do not pad
        # video = torch.Tensor(frames)

        # repeat last frame throughout padded frames
        # last_frame = video[len(frames)-1]
        # for i in range(len(frames), self.longest_video):
        #     video[i] = last_frame

        label = self.labels[index]
        return (video, label, len(frames))
