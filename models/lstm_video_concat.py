import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMVideoConcat(nn.Module):
    def __init__(self, feature_extractor, n_classes, n_visual_features=432, lstm_hidden_size=128):
        super(LSTMVideoConcat, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        # self.lstm = nn.GRU(input_size=n_classes, hidden_size=lstm_hidden_size, num_layers=3, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)

        self.criterion = nn.NLLLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch, timesteps, c, h, w = x.size()
        # x = x.view(batch*timesteps, c, h, w)

        print("x:", x.shape)
        visual_features = self.feature_extractor(x)
        print('visual features:', visual_features.shape)
        visual_features = visual_features.view(batch, timesteps, -1)

        output, (_, _) = self.lstm(visual_features)
        
        classes = self.classifier(output[:, -1, :])
        
        # Save all predictions
        # classes = self.classifier(output[:, :, :])
        return classes

        # If using NLL loss
        # softmax = F.log_softmax(classes, dim=1)
        # return softmax
