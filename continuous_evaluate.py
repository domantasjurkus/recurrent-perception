import torch

from datasets.xtion1continuous import Xtion1ContinuousDataset
from models.cifar_based import CifarBased
from models.cnn_lstm import CNNLSTMModel

FRAMES_PER_SEQUENCE=6
BATCH_SIZE = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Xtion1ContinuousDataset(root='../project-data/continuous_depth_keepaway', frames_per_sequence=FRAMES_PER_SEQUENCE)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)

def get_model():
    feature_extractor = CifarBased(n_classes=5)
    feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_depth_epoch10.pt'))

    model = CNNLSTMModel(feature_extractor, 5)
    model.load_state_dict(torch.load('saved_models/cnnlstmmodel_depth_epoch5.pt'))
    
    return model

model = get_model()
model.to(device)

with torch.no_grad():
    model.eval()
    for i, data in enumerate(loader):
        sequences, labels = data
        sequences, labels = sequences.to(device, dtype=torch.float), labels.to(device)

        outputs = model(sequences)
        _, predicted_indexes = torch.max(outputs.data, 1)

        print(sequences.shape, labels, predicted_indexes)
