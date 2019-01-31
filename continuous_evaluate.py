import torch

from datasets import Xtion1ContinuousDataset

FRAMES_PER_SEQUENCE=6

dataset = Xtion1ContinuousDataset(root='../project-data/continuous_depth_keepaway', frames_per_sequence=FRAMES_PER_SEQUENCE)

# todo