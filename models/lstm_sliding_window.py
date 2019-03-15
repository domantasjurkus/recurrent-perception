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

class LSTMSlidingWindow(nn.Module):
    def __init__(self, n_classes=5, n_visual_features=576, lstm_hidden_size=128, fps=6, device='cpu'):
        super(LSTMSlidingWindow, self).__init__()
        self.device = device
        self.fps = fps
        self.lstm_hidden_size = lstm_hidden_size
        
        self.feature_extractor = get_feature_extractor()
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.NLLLoss()
        self.device = device

    def init_hidden_and_cell(self):
        # hard-coded batch size of 1
        h0 = torch.zeros(1, 1, self.lstm_hidden_size, device=self.device)
        c0 = torch.zeros(1, 1, self.lstm_hidden_size, device=self.device)
        return (h0, c0)

    def get_classes(self, classifier_output, method=1):
        if method == 1:
            # 1) return prediction from last LSTM output BAD BAD BAD
            return classifier_output[:, -1, :]

        if method == 2:
            # 2) max-pool predictions over time
            classes, _ = classifier_output.max(1)
            return classes

        if method == 3:
            # 3) average over time
            n_cells = classifier_output.shape[1]
            classes = classifier_output.sum(dim=1) / n_cells
            return classes

        if method == 4:
            # 4) weight by g
            fps = classifier_output.shape[1]
            if fps > 1:
                g = torch.linspace(0, 1, fps, device=self.device)
            else:
                g = torch.tensor([1.], device=self.device)
            g = g.view(1, g.shape[0], 1)

            classifier_output = classifier_output * g
            classes = classifier_output.sum(dim=1)
            return classes
        
        raise Exception()

    def forward(self, video):
        batch, timesteps, c, h, w = video.size()
        c_in = video.view(batch*timesteps, c, h, w)
        c_out = self.feature_extractor(c_in)
        c_out = c_out.view(batch, timesteps, -1)

        hc = self.init_hidden_and_cell()

        # slide window with stride 1
        for i in range(self.fps, timesteps+1):
            feature_sequence = c_out[:, i-self.fps:i, :]
            lstm_out, hc = self.lstm(feature_sequence, hc)
            classifier_out = self.classifier(lstm_out[:, :, :])

            # pick 1 of 4 interpretations methods
            classes = self.get_classes(classifier_out, method=1)

        softmax = F.log_softmax(classes, dim=1)
        return softmax