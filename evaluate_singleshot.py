import torch

from datasets.xtion1video import Xtion1VideoDataset
from models.cifar_based import CifarBased

BATCH_SIZE = 1
N_CLASSES = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Xtion1VideoDataset(root='../project-data/continuous_depth_test')
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8)

def get_model():
    model = CifarBased(n_classes=5)
    model.load_state_dict(torch.load('saved_models/cifarbased_masked_epoch3_acc0.372212.pt'))

    return model

model = get_model()
model.to(device)
correct = 0

with torch.no_grad():
    model.eval()
    for i, data in enumerate(loader):
        batch, label, _ = data
        batch, label = batch.to(device, dtype=torch.float), label.to(device)
        print("video with label %s: " % label.item())
        
        # n_frames = video.shape[1]
        # n_sequences = n_frames // FRAMES_PER_SEQUENCE
        # bins = torch.tensor([[0] * N_CLASSES], dtype=torch.float).to(device)

        # for i in range(0, n_sequences):
        #     start_index = i*FRAMES_PER_SEQUENCE
        #     end_index = (i+1)*FRAMES_PER_SEQUENCE
        #     sequence = video[:, start_index:end_index, :, :, :]
            
        #     outputs = model(sequence)
        #     # print(outputs)
        #     bins += outputs

        #     # _, predicted_index = torch.max(outputs, 1)
        #     # predicted_index = predicted_index.item()
        #     # bins[predicted_index] += 1

        for video in batch:
            outputs = model(video)
            outputs = torch.exp(outputs)
            outputs = torch.sum(outputs, dim=0)
            print(outputs)
            _, predicted_id = outputs.max(0)
            print(predicted_id)
            predicted_id = predicted_id.cpu().numpy()
            print(predicted_id)

            if label.item() == predicted_id:
                correct += 1
    
    print("Total accuracy: %.1f%%" % (correct*100/dataset.__len__()))