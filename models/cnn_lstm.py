import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNLSTMModel(nn.Module):
    def __init__(self, feature_extractor, n_classes, lstm_hidden_size=128):
        super(CNNLSTMModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(input_size=n_classes, hidden_size=lstm_hidden_size, num_layers=3, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        samples, timesteps, c, h, w = x.size()
        c_in = x.view(samples*timesteps, c, h, w)
        c_out = self.feature_extractor(c_in)

        r_in = c_out.view(samples, timesteps, -1)
        
        # What I am doing
        r_out, _ = self.lstm(r_in)
        
        # What I feel I should be doing
        # r_out, HIDDEN_OUTPUT = self.lstm(r_in, HIDDEN_OUTPUT)

        # Save prediction from last LSTM output
        classes = self.classifier(r_out[:, -1, :])
        softmax = F.log_softmax(classes, dim=1)
        return softmax