import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

def init_hidden_and_cell(hidden_size=128, device="cuda:0"):
    # hard-coded batch size of 1
    h0 = torch.zeros(1, 1, hidden_size, device=device)
    c0 = torch.zeros(1, 1, hidden_size, device=device)
    return (h0, c0)

class LSTMFullVideoNaive(nn.Module):
    def __init__(self, n_classes=5, n_visual_features=576, lstm_hidden_size=128):
        super(LSTMFullVideoNaive, self).__init__()
        self.hc0 = init_hidden_and_cell()

        self.feature_extractor = get_feature_extractor()
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.criterion = nn.NLLLoss()

    def forward(self, video, output_cell=-1):
        # batch always 1
        batch, timesteps, c, h, w = video.size()
        c_in = video.view(batch*timesteps, c, h, w)

        # c_in = (50, 1, 480, 640)

        visual_features = self.feature_extractor(c_in)
        visual_features = visual_features.view(batch, timesteps, -1)

        # (1, timesteps, 576)
        
        output, _ = self.lstm(visual_features, self.hc0)

        classes = self.classifier(output[:, output_cell, :])
        softmax = F.log_softmax(classes, dim=1)
        return softmax