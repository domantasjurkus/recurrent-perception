import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CifarBased(nn.Module):
    def __init__(self, n_classes):
        super(CifarBased, self).__init__()
        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(12, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(24, 24, kernel_size=3),
            nn.Conv2d(24, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.5),
        )

        self.classifier = nn.Linear(576, self.n_classes)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classes = self.classifier(features)
        return classes
        # return F.log_softmax(classes, dim=1)
