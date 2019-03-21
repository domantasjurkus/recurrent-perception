import numpy as np
import torch
import torch.nn.functional as F

from datasets.xtion1video import Xtion1VideoDataset
from models.lstm_sliding_window import LSTMSlidingWindow

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 1
N_CLASSES = 5
FPS = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = Xtion1VideoDataset(root='../project-data/continuous_depth_test')
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

def get_model(device):
    model = LSTMSlidingWindow(device=device)
    model.load_state_dict(torch.load('saved_models/lstmslidingwindow_masked_epoch1_acc0.360000.pt'))
    
    print(model.state_dict().keys())

    return model

model = get_model(device)
model.to(device)
correct = 0

class_correct = [0]*N_CLASSES
class_total = [0]*N_CLASSES
total = 0
correct = 0

# with torch.no_grad():
#     model.eval()
#     for i, data in enumerate(loader):
#         batch, label, _ = data
#         batch, label = batch.to(device, dtype=torch.float), label.to(device)
#         print("video with label %s: " % label.item())
        
#         # for each video: classify from 6 to n frames and observe how accuracy changes
#         _, video_length, _, _, _ = batch.shape
#         for i in range(FPS, video_length):
#             video_subsample = batch[:, 0:i, :, :, :]
#             log_softmax_outputs = model(video_subsample)
#             softmax_outputs = F.softmax(log_softmax_outputs)
#             # print(softmax_outputs, end=' ')
#             _, predicted_index = log_softmax_outputs.max(1)
#             # print(1 if predicted_index[0].item() == label.item() else 0)

#         # sum up total correct
#         batch_size = label.size(0)
#         total += batch_size
#         correct_vector = (predicted_index == label)
#         correct += correct_vector.sum().item()

#         # sum up per-class correct
#         for j in range(len(label)):
#             target = label[j]
#             class_correct[target] += correct_vector[j].item()
#             class_total[target] += 1
    
#         break
            
    
#     print("Total accuracy: %.1f%%" % (correct*100/dataset.__len__()))

def test(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    running_loss = 0.0
    total_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    minibatch_count = len(test_loader)
    print('testing minibatch count:', minibatch_count)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            videos, targets, _ = data
            videos, targets = videos.to(device, dtype=torch.float), targets.to(device)

            outputs = model(videos)

            loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY
            total_loss += loss.item()
            running_loss += loss.item()
            
            # pick index with max activation value
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            # bin predictions into confusion matrix
            for j in range(len(videos)):
                actual = targets[j].item()
                predicted = predicted_indexes[j].item()
                confusion[actual][predicted] += 1

            # sum up total correct
            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indexes == targets)
            correct += correct_vector.sum().item()
            
            # sum up per-class correct
            for j in range(len(targets)):
                target = targets[j]
                class_correct[target] += correct_vector[j].item()
                class_total[target] += 1

            if i % 5 == 0:
                print('minibatches: %d/%d running loss: %.3f' % (i, minibatch_count, running_loss))
                running_loss = 0.0
        

    print('Total test samples: ', class_total)
    print('Correct predictions:', class_correct)
    acc = correct / total
    print('Test accuracy: %f' % acc)
    print('Per-class accuracy:', np.asarray(class_correct)/np.asarray(class_total))
    print(confusion)
    print('Total testing loss:', total_loss)
    return acc

test(model, loader, N_CLASSES, 1.0, device=device)