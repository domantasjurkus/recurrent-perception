import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def l1():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, bias=False),
        nn.MaxPool2d()
    )

def l2():
    return nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, bias=False),
        nn.MaxPool2d()
    )

def l3():
    return nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False),
        nn.MaxPool2d()
    )

def l4():
    return nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, bias=False),
        nn.MaxPool2d()
    )

class UPCFeatureExtractor(nn.Module):
    def __init__(self, n_classes=4):
        super(UPCFeatureExtractor, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            l1(),
            l2(),
            l3(),
            l4()
        )
        self.classifier = nn.Linear(1210944, n_classes)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classes = self.classifier(features)
        # return F.log_softmax(classes, dim=1)
        return classes