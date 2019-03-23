import torch

from datasets.xtion1video import Xtion1VideoDataset
from models.lstm_sliding_window import LSTMSlidingWindow
from train_test_sliding_window import *

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

BATCH_SIZE = 1
SHUFFLE = True
FRAMES_PER_SEQUENCE = 6

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

dataset_train = Xtion1VideoDataset(root='../project-data/continuous_masked')
dataset_test = Xtion1VideoDataset(root='../project-data/continuous_masked_test')

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
    # feature_extractor = CifarBased(n_classes=n_classes)
    # feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_masked_epoch3_acc0.372212.pt'))

    # model = LSTMSnippet(feature_extractor.features, n_classes)
    model = LSTMSlidingWindow(n_classes, device=device)
    return model

model = get_model()
model.to(device)

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=100, save=False, masked=True, device=device)
    # train(model, loader_train, loader_test, n_classes, epochs=100, save=False, masked=True, device=device, fps=FRAMES_PER_SEQUENCE)

    # test(model, loader_keepaway, n_classes, 1, device=device)
