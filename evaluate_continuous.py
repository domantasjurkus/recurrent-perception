import torch
import random

# random.seed(1337)
# torch.manual_seed(1337)

from datasets.xtion1video import Xtion1VideoDataset
from models.cifar_based import CifarBased
from models.lstm_video import LSTMVideo
from train_test_continuous_evaluate import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

BATCH_SIZE = 1
SHUFFLE = False

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

dataset_train = Xtion1VideoDataset(root='../project-data/continuous_masked')
dataset_test = Xtion1VideoDataset(root='../project-data/continuous_masked_test')

train_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": SHUFFLE,
}

test_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": 8,
    "shuffle": SHUFFLE,
}

loader_train = torch.utils.data.DataLoader(dataset_train, **train_params)
loader_test = torch.utils.data.DataLoader(dataset_test, **test_params)

def get_model_for_evaluation():
    feature_extractor = CifarBased(n_classes=n_classes)
    feature_extractor.load_state_dict(torch.load('saved_models/cifarbased_nodrop_masked_epoch6_acc35.941499.pt'))

    model = LSTMVideo(feature_extractor.features, n_classes)
    model.load_state_dict(torch.load('saved_models/video_lstmvideo_masked_epoch53.pt'))
    
    return model

model = get_model_for_evaluation()
model.to(device)
model.eval()

def test_evaluate(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, lstm_output_cell, device="cpu"):
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    running_loss = 0.0
    total_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int)

    # minibatch_count = len(test_loader)
    # print('testing minibatch count:', minibatch_count)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sequences, targets, video_lengths = data
            sequences, targets = sequences.to(device, dtype=torch.float), targets.to(device)

            outputs = model(sequences, video_lengths, lstm_output_cell)

            # Print per-cell predictions for correctly classified videos:
            # last_output = outputs[:, video_lengths.item()-1, :]
            # _, all_indices = torch.max(outputs.data, 2)
            # _, predicted_index = torch.max(last_output.data, 1)
            # if predicted_index == targets:
                # print("target and seq_len:", targets[0].item(), video_lengths[0].item(), end=' ')
                # print(list(all_indices[0].cpu().numpy()))

            # pick last output
            # outputs = outputs[:, video_lengths.item()-1, :]

            loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY
            total_loss += loss.item()
            running_loss += loss.item()
            
            _, predicted_indices = torch.max(outputs.data, 1)
            
            # bin predictions into confusion matrix
            for j in range(len(sequences)):
                actual = targets[j].item()
                predicted = predicted_indices[j].item()
                confusion[actual][predicted] += 1

            # sum up total correct
            batch_size = targets.size(0)
            total += batch_size
            correct_vector = (predicted_indices == targets)
            correct += correct_vector.sum().item()
            
            # sum up per-class correct
            for j in range(len(targets)):
                target = targets[j]
                class_correct[target] += correct_vector[j].item()
                class_total[target] += 1

            if i % 5 == 0:
                # print('minibatches: %d/%d running loss: %.3f' % (i, minibatch_count, running_loss))
                running_loss = 0.0
        
        testing_losses.append(total_loss)

    # print('Total test samples: ', class_total)
    # print('Correct predictions:', class_correct)
    per_class = list(np.asarray(class_correct)/np.asarray(class_total))
    print('Per-class accuracy:', per_class, end=' ')
    total_accuracy = correct / total
    return total_accuracy
    # accuracies.append(total_accuracy)
    # print('Test accuracy: %d %%' % total_accuracy)
    # print(confusion)
    
    # print('Total testing loss:', total_loss)


if __name__ == '__main__':
    # varying output cell
    # for lstm_output_cell in range(5,90,1):
    #     print("LSTM output cell:", lstm_output_cell, end='\t')
    #     acc = test_evaluate(model, loader_test, n_classes, 1, lstm_output_cell, device)
    #     print('Total acc:', acc)

    # last (non-padded) cell only
    acc = test_evaluate(model, loader_test, n_classes, 1, -1, device)

