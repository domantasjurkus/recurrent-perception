import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# for 640x480
def get_feature_extractor():
    return nn.Sequential(
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
    )


class CifarBased(nn.Module):
    def __init__(self, n_classes):
        super(CifarBased, self).__init__()
        self.n_classes = n_classes

        self.features = get_feature_extractor()

        self.classifier = nn.Linear(576, self.n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.NLLLoss()

    def forward(self, batch):
        features = self.features(batch)
        features = features.view(features.shape[0], -1)

        classes = self.classifier(features)
        softmax = F.log_softmax(classes, dim=1)
        return softmax
