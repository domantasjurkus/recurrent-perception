import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSnippet(nn.Module):
    def __init__(self, feature_extractor, n_classes, n_visual_features=576, lstm_hidden_size=128):
        super(LSTMSnippet, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        # freeze parameters for debugging - model should perform much worse
        # for param in self.feature_extractor.parameters():
        #     try:
        #         param.requires_grad = False
        #     except:
        #         print("warn: not freezing ", param)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch, timesteps, c, h, w = x.size()
        c_in = x.view(batch*timesteps, c, h, w)
        c_out = self.feature_extractor(c_in)

        r_in = c_out.view(batch, timesteps, -1)
        
        output, _ = self.lstm(r_in)

        # Save prediction from last LSTM output
        classes = self.classifier(output[:, -1, :])
        # softmax = F.log_softmax(classes, dim=1)
        # return softmax
        return classes