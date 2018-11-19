import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simple import conv1, conv2, conv3

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=1024, n_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.conv1 = conv1()
        self.conv2 = conv2()
        self.conv3 = conv3()
        
        self.i2h = nn.Linear(298304 + hidden_size, hidden_size) 
        self.fc = nn.Linear(298304 + hidden_size, n_classes)

    def forward(self, inp, hidden):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        features = c3.view(c3.size(0), -1)
        # print("Extracted feature shape:", features.shape)
        
        combined = torch.cat((features, hidden), 1)
        combined = combined.data
        # print("combined shape:", combined.shape)
        hidden = self.i2h(combined)
        hidden = hidden.data

        classes = self.fc(combined)
        softmaxed = F.log_softmax(classes, dim=1)
        return softmaxed, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)