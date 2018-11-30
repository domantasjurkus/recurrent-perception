import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def conv1():
    return nn.Sequential(
        nn.Conv2d(1, 64, 7, 2, 0, bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )

def conv2():
    return nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

def conv3():
    return nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

class SimpleNetwork(nn.Module):
    def __init__(self, n_classes=4):
        super(SimpleNetwork, self).__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(
            conv1(),
            conv2(),
            conv3()
        )
        self.classifier = nn.Linear(1210944, n_classes)

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.NLLLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.Adam(model.parameters())
        # criterion = nn.NLLLoss()

    def forward(self, inp):
        # c1 = self.conv1(inp)
        # c2 = self.conv2(c1)
        # c3 = self.conv3(c2)
        c3 = self.features(inp)
        features = c3.view(c3.size(0), -1)
        classes = self.classifier(features)
        return F.log_softmax(classes, dim=1)