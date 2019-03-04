import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
sys.path.append(os.getcwd()+'/models')
print(sys.path)

def get_pre_concat_features():
    return nn.Sequential(
        nn.Conv2d(1, 12, kernel_size=11, stride=4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(12, 24, kernel_size=5, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def get_post_concat_features():
    return nn.Sequential(
        # Implicit hardcode to 6 frames (144 channels)
        nn.Conv2d(144, 144, kernel_size=3),
        nn.Conv2d(144, 144, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

class SnippetConcat(nn.Module):
    def __init__(self, n_classes):
        super(SnippetConcat, self).__init__()

        self.pre_concat_features = get_pre_concat_features()
        self.post_concat_features = get_post_concat_features()
        
        self.classifier = nn.Sequential(
            nn.Linear(3456, 512),
            nn.Linear(512, n_classes),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.NLLLoss()

    def features(self, x):
        batch, timesteps, c, h, w = x.size()
        x = x.view(batch*timesteps, c, h, w)

        pre = self.pre_concat_features(x)
        batch_times_frames, f_height, f_width, f_depth = pre.shape
        pre = pre.view(batch, timesteps, f_height, f_width, f_depth)
        pre = pre.view(batch, timesteps*f_height, f_width, f_depth) # Concatenate features in a single batch
        post = self.post_concat_features(pre)

        post = post.view(batch, -1)
        return post

    def forward(self, x):
        post = self.features(x)

        classes = self.classifier(post)

        softmax = F.log_softmax(classes, dim=1)
        # print(F.softmax(classes, dim=1))
        return softmax