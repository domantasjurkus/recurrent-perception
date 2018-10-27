import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):
    def __init__(self, n_classes=2):
        super(SimpleNetwork, self).__init__()
        self.conv = nn.Conv2d(1, 3, 5, 1)
        self.fc = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 612)
        x = self.fc(x)
        return x
