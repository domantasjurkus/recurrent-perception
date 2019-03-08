import torch

from datasets.xtion1video import Xtion1VideoDataset
from models.snippet_concat import SnippetConcat

FRAMES_PER_SEQUENCE = 6
BATCH_SIZE = 1
N_CLASSES = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Xtion1VideoDataset(root='../project-data/continuous_depth_test')
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)

def get_model():
    # feature_extractor = CifarBased(n_classes=N_CLASSES)
    # feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_depth_epoch10.pt'))

    # model = LSTMSnippet(feature_extractor, N_CLASSES)
    model = SnippetConcat(N_CLASSES)
    model.load_state_dict(torch.load('saved_models/snippetconcat_masked_epoch5_acc0.369515.pt'))
    
    return model

model = get_model()
model.to(device)
correct = 0

with torch.no_grad():
    model.eval()
    for i, data in enumerate(loader):
        video, label, _ = data
        video, label = video.to(device, dtype=torch.float), label.to(device)
        print("Predictions for video with label %s: " % label.item(), end='')
        
        n_frames = video.shape[1]
        n_sequences = n_frames // FRAMES_PER_SEQUENCE
        bins = torch.tensor([[0] * N_CLASSES], dtype=torch.float).to(device)

        for i in range(0, n_sequences):
            start_index = i*FRAMES_PER_SEQUENCE
            end_index = (i+1)*FRAMES_PER_SEQUENCE
            sequence = video[:, start_index:end_index, :, :, :]
            
            outputs = model(sequence)
            bins += outputs

            # _, predicted_index = torch.max(outputs, 1)
            # predicted_index = predicted_index.item()
            # bins[predicted_index] += 1
        
        max_val, max_idx = bins.max(1)
        predicted_id = max_idx.cpu().numpy()[0]
        print(bins[0].cpu().numpy())
        print(predicted_id, end=" ")

        if label.item() == predicted_id:
            correct += 1
    
    print("Total accuracy: %.1f%%" % (correct*100/dataset.__len__()))