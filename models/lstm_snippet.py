import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class LSTMSnippet(nn.Module):
    def __init__(self, n_classes=5, n_visual_features=576, lstm_hidden_size=128, device='cpu'):
        super(LSTMSnippet, self).__init__()
        self.feature_extractor = get_feature_extractor()
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.NLLLoss()
        self.device = device

    def get_classes(self, lstm_output):
        # 
        # pick 1 of 4 interpretations methods
        # 
        # 1) return prediction from last LSTM output BAD BAD BAD
        classes = lstm_output[:, -1, :]

        # 2) max-pool predictions over time
        # classes, _ = lstm_output.max(1)

        # 3) sum over time
        # classes = lstm_output.sum(dim=1)

        # 4) weight by g
        # fps = lstm_output.shape[1]
        # g = torch.linspace(0, 1, fps, device=self.device)
        # g = g.view(1, g.shape[0], 1)
        # lstm_output = lstm_output * g
        # classes = lstm_output.sum(dim=1)

        return classes

    def forward(self, x):
        batch, timesteps, c, h, w = x.size()
        c_in = x.view(batch*timesteps, c, h, w)
        c_out = self.feature_extractor(c_in)

        r_in = c_out.view(batch, timesteps, -1)
        
        output, _ = self.lstm(r_in)
        lstm_output = self.classifier(output[:, :, :])
        
        classes = self.get_classes(lstm_output)

        softmax = F.log_softmax(classes, dim=1)
        return softmax