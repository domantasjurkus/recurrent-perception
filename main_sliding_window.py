import numpy as np
import torch
import torch.nn.functional as F

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
STRIDE = 1

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
    model = LSTMSlidingWindow(n_classes, device=device)
    # model.load_state_dict(torch.load('saved_models/lstmslidingwindow_masked_epoch14_acc0.400000.pt'))
    return model

model = get_model()
model.to(device)

training_losses = []
testing_losses = []
accuracies = []

def init_hidden_and_cell(hidden_size=128, device="cpu"):
    # hard-coded batch size of 1
    h0 = torch.zeros(1, 1, hidden_size, device=device)
    c0 = torch.zeros(1, 1, hidden_size, device=device)
    return (h0, c0)

def train(model, train_loader, test_loader, n_classes, epochs=10, masked=False, save=False, hidden_size=128, fps=6, device="cpu"):
    minibatch_count = len(train_loader)
    print('training minibatch count:', minibatch_count)
    print("device:", device)

    TEST_LOSS_MULTIPLY = len(train_loader)/len(test_loader)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            videos, targets, _ = data
            videos, targets = videos.to(device, dtype=torch.float), targets.to(device)
            model.optimizer.zero_grad()

            batch, timesteps, _, _, _ = videos.size()

            hc = init_hidden_and_cell(hidden_size, device)

            for end in range(fps, timesteps+1, STRIDE):
                feature_sequence = videos[:, end-fps:end, :]
                outputs, hc = model(feature_sequence, hc)
                hc = (hc[0].detach(), hc[1].detach())

                # backpropagate for each snippet
                loss = model.criterion(outputs, targets)
                loss.backward()
                model.optimizer.step()

                total_loss += loss.item()
                running_loss += loss.item()

            if i % 5 == 0:
                print('epoch: %d minibatches: %d/%d loss: %.3f' % (epoch, i, minibatch_count, running_loss))
                running_loss = 0.0
        
        training_losses.append(total_loss)
        acc = test(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device=device)
        print('Total training loss:', total_loss)

        if save:
            save_model(model, masked, epoch, acc)

        print('Training losses:', str(training_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Testing losses:', str(testing_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Accuracies:', str(accuracies).replace(",", "").replace("[", "").replace("]", ""))

    print('Finished Training')

def save_model(model, masked, epoch, acc):
    model_name = type(model).__name__.lower()
    is_masked = 'masked' if masked else 'depth'
    torch.save(model.state_dict(), "saved_models/%s_%s_epoch%d_acc%f.pt" % (model_name, is_masked, epoch, acc))
    print("model saved")

def test(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, fps=6, device="cpu"):
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
            batch_size, timesteps, _, _, _ = videos.size()

            hc = init_hidden_and_cell(device=device)
            # prediction_bins = torch.zeros([1, n_classes], device=device)

            # slide window with STRIDE
            for k in range(fps, timesteps+1, STRIDE):
                feature_sequence = videos[:, k-fps:k, :]
                outputs, hc = model(feature_sequence, hc)

                softmax_predictions = F.softmax(outputs, dim=1)
                # prediction_bins += softmax_predictions

                # backpropagate for each snippet
                loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY

                total_loss += loss.item()
                running_loss += loss.item()

            #TODO: extend with accuracy metric (correctly classified snippets / number of snippets)
            #TODO: the same for single shot, but correctly classified frames/ number of frames

            # pick index with max activation value
            # _, predicted_indexes = torch.max(prediction_bins.data, 1)

            # outputs of the last block
            _, predicted_indexes = torch.max(outputs, 1)
            
            # bin predictions into confusion matrix
            for j in range(batch_size):
                actual = targets[j].item()
                predicted = predicted_indexes[j].item()
                confusion[actual][predicted] += 1

            # sum up total correct
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
        
        testing_losses.append(total_loss)

    print('Total test samples: ', class_total)
    print('Correct predictions:', class_correct)
    acc = correct / total
    accuracies.append(acc)
    print('Test accuracy: %f' % acc)
    print('Per-class accuracy:', np.asarray(class_correct)/np.asarray(class_total))
    print(confusion)
    print('Total testing loss:', total_loss)
    return acc

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=50, save=True, masked=True, device=device)