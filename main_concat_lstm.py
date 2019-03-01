import torch

from datasets.xtion1video import Xtion1VideoDataset
from models.snippet_concat import SnippetConcat # Features
from models.lstm_video_concat import LSTMVideoConcat # LSTM + classifier

from train_test_continuous_concat import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

BATCH_SIZE = 1
SHUFFLE = True
FRAMES_PER_SNIPPET = 6

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

dataset_train = Xtion1VideoDataset(root='../project-data/continuous_masked_resized')
dataset_test = Xtion1VideoDataset(root='../project-data/continuous_masked_test_resized')

train_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": True,
}

test_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": True,
}

loader_train = torch.utils.data.DataLoader(dataset_train, **train_params)
loader_test = torch.utils.data.DataLoader(dataset_test, **test_params)

def get_model():
    feature_extractor = SnippetConcat(n_classes=n_classes)

    model = LSTMVideoConcat(feature_extractor, n_classes)

    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=100, save=False, masked=True, device=device)

    # test(model, loader_keepaway, n_classes, 1, device=device)
