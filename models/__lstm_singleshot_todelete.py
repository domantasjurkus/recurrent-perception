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

class LSTMSingleshot(nn.Module):
    def __init__(self, n_classes=5, n_visual_features=576, lstm_hidden_size=128):
        super(LSTMSingleshot, self).__init__()
        self.n_classes = n_classes

        self.feature_extractor = get_feature_extractor()
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, self.n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.criterion = nn.NLLLoss()

    def forward(self, x, hc):
        # batch always 1
        timesteps, c, h, w = x.size()
        # c_in = x.view(timesteps, c, h, w)
        # c_in = (variable, 1, 480, 640)

        visual_features = self.feature_extractor(x)
        # flatten visual features (could also fully-connect here, but does not seem to make sense)
        # hard-coded batch of 1
        visual_features = visual_features.view(1, timesteps, -1)
        
        lstm_output, hc = self.lstm(visual_features, hc)
        # lstm_output, _ = self.lstm(visual_features)
        lstm_output = lstm_output.view(1, -1)
        classes = self.classifier(lstm_output)

        softmax = F.log_softmax(classes, dim=1)
        return softmax, hc