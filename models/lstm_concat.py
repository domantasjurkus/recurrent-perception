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

        output, _ = self.lstm(feature_aggregate)
                
        classes = self.classifier(output[:, -1, :]) # bad bad boy

        softmax = F.log_softmax(classes, dim=1)
        return softmax
