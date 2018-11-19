import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simple import conv1, conv2, conv3

# before writing this model, make sure that frame data is sequential (more dataloader work)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=1024, n_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # feature extraction part
        self.conv1 = conv1()
        self.conv2 = conv2()
        self.conv3 = conv3()

        # classification part
        self.lstm = nn.LSTM(n_features, hidden_size)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        
        # self.i2h = nn.Linear(298304 + hidden_size, hidden_size)
        self.fc = nn.Linear(298304 + hidden_size, n_classes)

    def forward(self, inp, hidden):
        # feature extraction
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        features = c3.view(c3.size(0), -1)
        # print("Extracted feature shape:", features.shape)
        
        # classification
        # 
        # to do tomorrow
        # 
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