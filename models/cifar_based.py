import torch.nn as nn
import torch.nn.functional as F

class CifarBased(nn.Module):
    def __init__(self, n_classes=2):
        super(CifarBased, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 11, 4)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(3, 3, 5, 2)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3) # maybe scrap if overfitting?
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(612, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 612)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
