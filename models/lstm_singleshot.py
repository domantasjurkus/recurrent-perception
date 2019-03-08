import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This model is rather naive: it takes the whole video and predicts
# only on the last cell (or some other cell if specified)

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

class LSTMSingleshot(nn.Module):
    # def __init__(self, feature_extractor, n_classes, n_visual_features=576, lstm_hidden_size=128):
    def __init__(self, n_classes, n_visual_features=576, lstm_hidden_size=128):
        super(LSTMSingleshot, self).__init__()
        self.feature_extractor = get_feature_extractor()
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.criterion = nn.NLLLoss()

    def forward(self, x, video_lengths, output_cell=-1):
        samples, timesteps, c, h, w = x.size()
        c_in = x.view(samples*timesteps, c, h, w)

        visual_features = self.feature_extractor(c_in)
        visual_features = visual_features.view(samples, timesteps, -1)

        output, _ = self.lstm(visual_features)

        classes = self.classifier(output[:, output_cell, :])
        softmax = F.log_softmax(classes, dim=1)
        return softmax