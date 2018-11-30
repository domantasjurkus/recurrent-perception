import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CifarBased(nn.Module):
    def __init__(self, n_classes=4):
        super(CifarBased, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(3, 3, kernel_size=3),
            nn.Conv2d(3, 3, kernel_size=3),
            nn.Conv2d(3, 3, kernel_size=3), # maybe scrap if overfitting?
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Linear(45, n_classes)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 612)

        features = self.features(x)
        features = features.view(features.size(0), -1)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.classifier(features)
        return x
