import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This model will make precition per concatenated feature vector, aggregate probabilities
# and output a full video prediction

class LSTMConcat(nn.Module):
    def __init__(self, feature_extractor, n_classes, frames_per_sequence, n_visual_features=3456, lstm_hidden_size=256, device="cpu"):
        super(LSTMConcat, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.frames_per_sequence = frames_per_sequence
        self.n_visual_features = n_visual_features
        self.feature_extractor = feature_extractor

        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)        
        self.classifier = nn.Linear(256, self.n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)

        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()

    def features(self, x):
        batch, timesteps, c, h, w = x.size()
        x = x.view(batch*timesteps, c, h, w)

        pre = self.pre_concat_features(x)
        batch_times_frames, f_height, f_width, f_depth = pre.shape
        pre = pre.view(batch, timesteps, f_height, f_width, f_depth)
        pre = pre.view(batch, timesteps*f_height, f_width, f_depth) # Concatenate features in a single batch
        post = self.post_concat_features(pre)

        post = post.view(batch, -1)
        return post

    # The 3rd argument (_) is video_lengths that is norally passed to LSTMSingleshot
    # to observe the effects of classifying based on varying the LSTM output cell.
    # We do not use it here.

    # two ways to classify: look at the last LSTM cell
    # or aggregate probabilities over all cells
    def forward(self, x, _):
        batch, timesteps, c, h, w = x.size()
        # x = x.view(batch*timesteps, c, h, w)

        # use frames_per_sequence to split input video in chunks, then ...
        # predictions = [0.0]*self.n_classes

        n_sequences = timesteps // self.frames_per_sequence

        feature_aggregate = torch.zeros((batch, n_sequences, self.n_visual_features), device=self.device)

        for i in range(0, n_sequences):
            sequence_start = i*self.frames_per_sequence
            sequence_end = (i+1)*self.frames_per_sequence
            sequence = x[:, sequence_start:sequence_end, :, :, :]
            sequence_features = self.feature_extractor.features(sequence)
            feature_aggregate[:, i] = sequence_features

        # visual_features = visual_features.view(batch, timesteps, -1)

        output, _ = self.lstm(feature_aggregate)
                
        classes = self.classifier(output[:, -1, :]) # bad bad boy
        return classes

        # If using NLL loss
        # softmax = F.log_softmax(classes, dim=1)
        # return softmax
