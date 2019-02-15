import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMVideo(nn.Module):
    def __init__(self, feature_extractor, n_classes, n_visual_features=576, lstm_hidden_size=128):
        super(LSTMVideo, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(input_size=n_visual_features, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        # self.lstm = nn.GRU(input_size=n_classes, hidden_size=lstm_hidden_size, num_layers=3, batch_first=True)
        
        self.classifier = nn.Linear(128, n_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        # freeze parameters for debugging - model should perform much worse
        # for param in self.feature_extractor.parameters():
        #     try:
        #         param.requires_grad = False
        #     except:
        #         print("warn: not freezing ", param)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, video_lengths):
        samples, timesteps, c, h, w = x.size()
        c_in = x.view(samples*timesteps, c, h, w)

        visual_features = self.feature_extractor(c_in)
        visual_features = visual_features.view(samples, timesteps, -1)

        # SORT YOUR TENSORS BY LENGTH!
        # https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e#file-pad_packed_demo-py-L26
        video_lengths, perm_idx = video_lengths.sort(0, descending=True)
        visual_features = visual_features[perm_idx]
        
        packed_input = pack_padded_sequence(visual_features, video_lengths.cpu().numpy(), batch_first=True)
        packed_output, _ = self.lstm(packed_input)

        # unpack your output if required
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Save prediction from last LSTM output
        classes = self.classifier(output[:, -1, :])
        # softmax = F.log_softmax(classes, dim=1)
        # return softmax
        return classes