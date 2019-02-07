import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CifarBased(nn.Module):
    def __init__(self, n_classes):
        super(CifarBased, self).__init__()
        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(12, 24, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(24, 36, kernel_size=3),
            # nn.Conv2d(36, 36, kernel_size=3),
            # nn.Conv2d(3, 3, kernel_size=3), # maybe scrap if overfitting?
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Dropout(p=0.3),
        )

        self.classifier = nn.Linear(7344, self.n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classes = self.classifier(features)
        return classes
        # return F.log_softmax(classes, dim=1)
