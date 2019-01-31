import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 
# Adapted  from:
# https://www.semanticscholar.org/paper/Robot-Aided-Cloth-Classification-Using-Depth-and-Gabas-Corona/05db2f27297305f757be716261886898c38a7920
# without dropout layers
# 

def l1():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
    )

def l2():
    return nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5, bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
    )

def l3():
    return nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
    )

def l4():
    return nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, bias=False),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
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
        self.classifier = nn.Linear(47952, n_classes)
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classes = self.classifier(features)
        return classes