import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.optim as optim

from models.simple import conv1

class ResnetBased(nn.Module):
    def __init__(self, n_classes=4):
        super(ResnetBased, self).__init__()

        # resnet18.modules = [layer1, layer2, layer3, layer4, avgpool, fc]
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Change from 3 channels to 1
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        
        # https://discuss.pytorch.org/t/why-torchvision-models-can-not-forward-the-input-which-has-size-of-larger-than-430-430/2067/9
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)

        self.optimizer = optim.Adam(self.model.fc.parameters())
        self.criterion = nn.NLLLoss()

    def forward(self, inp):
        classes = self.model(inp)
        return F.log_softmax(classes, dim=1)

    def features(self):
        return self.model.avgpool

    def classifier(self):
        return self.model.fc