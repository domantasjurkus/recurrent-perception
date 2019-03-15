import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def get_pre_concat_features():
    return nn.Sequential(
        nn.Conv2d(1, 12, kernel_size=11, stride=4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(12, 24, kernel_size=5, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def get_post_concat_features():
    return nn.Sequential(
        # Implicit hardcode to 6 frames (144 channels)
        nn.Conv2d(144, 144, kernel_size=3),
        nn.Conv2d(144, 144, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

class LSTMConcat(nn.Module):
    # def __init__(self, feature_extractor, n_classes, frames_per_sequence, n_visual_features=3456, lstm_hidden_size=256, device="cpu"):
    def __init__(self, n_classes, frames_per_sequence, n_visual_features=3456, lstm_hidden_size=256, device="cpu"):
        super(LSTMConcat, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.frames_per_sequence = frames_per_sequence
        self.n_visual_features = n_visual_features

        self.pre_concat_features = get_pre_concat_features()
        self.post_concat_features = get_post_concat_features()

        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, self.n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)

        self.criterion = nn.NLLLoss()

    def features(self, sequence):
        batch, timesteps, c, h, w = sequence.size()
        sequence = sequence.view(batch*timesteps, c, h, w)

        pre = self.pre_concat_features(sequence)
        batch_times_frames, f_height, f_width, f_depth = pre.shape
        pre = pre.view(batch, timesteps, f_height, f_width, f_depth)
        pre = pre.view(batch, timesteps*f_height, f_width, f_depth) # Concatenate features in a single batch
        post = self.post_concat_features(pre)

        post = post.view(batch, -1)
        return post
    
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

    # two ways to classify: look at the last LSTM cell
    # or aggregate probabilities over all cells
    def forward(self, video, _):
        batch, timesteps, c, h, w = video.size()

        n_sequences = timesteps // self.frames_per_sequence

        feature_aggregate = torch.zeros((batch, n_sequences, self.n_visual_features), device=self.device)

        for i in range(0, n_sequences):
            sequence_start = i*self.frames_per_sequence
            sequence_end = (i+1)*self.frames_per_sequence
            sequence = video[:, sequence_start:sequence_end, :, :, :]
            sequence_features = self.features(sequence)
            feature_aggregate[:, i] = sequence_features

        lstm_output, _ = self.lstm(feature_aggregate)
        classifier_output = self.classifier(lstm_output[:, :, :])
                
        # pick 1 of 4 interpretations methods
        classes = self.get_classes(classifier_output, method=3)

        softmax = F.log_softmax(classes, dim=1)
        return softmax
