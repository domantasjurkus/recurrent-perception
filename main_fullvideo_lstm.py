import torch
import random
import numpy as np

from datasets.xtion1video import Xtion1VideoDataset
from models.cifar_based import CifarBased
from models.lstm_fullvideo import LSTMFullVideo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 1
SHUFFLE = True

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

dataset_train = Xtion1VideoDataset(root='data/continuous_masked')
dataset_test = Xtion1VideoDataset(root='data/continuous_masked_test')

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)

def get_model():
    model = LSTMFullVideo()
    return model

model = get_model()
model.to(device)

training_losses = []
testing_losses = []
accuracies = []

def model_outputs_to_classes(classifier_output, method=1):
    if method == 1:
        # 1) return prediction from last LSTM output BAD BAD BAD
        return classifier_output[:, -1, :]

    if method == 2:
        # 2) max-pool predictions over time
        classes, _ = classifier_output.max(1)
        return classes

    if method == 3:
        # 3) average over time
        n_cells = classifier_output.shape[1]
        classes = classifier_output.sum(dim=1) / n_cells
        return classes

    if method == 4:
        # 4) weight by g
        fps = classifier_output.shape[1]
        if fps > 1:
            g = torch.linspace(0, 1, fps, device=device)
        else:
            g = torch.tensor([1.], device=device)
        g = g.view(1, g.shape[0], 1)

        classifier_output = classifier_output * g
        classes = classifier_output.sum(dim=1)
        return classes
    
    raise Exception()

def train(model, train_loader, test_loader, n_classes, epochs=10, masked=False, save=False, device="cpu"):
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
            # batch_size, timesteps, _, _, _ = videos.shape
            model.optimizer.zero_grad()
            
            outputs = model(videos)
            outputs = model_outputs_to_classes(outputs, method=1)

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
        # test_per_frame(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device=device)
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
            batch_size, timesteps, _, _, _ = videos.shape

            outputs = model(videos)
            outputs = model_outputs_to_classes(outputs, method=1)

            loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY
            total_loss += loss.item()
            running_loss += loss.item()
            
            # pick index with max activation value to indicate class
            _, predicted_indexes = torch.max(outputs.data, 1)
            
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
            for j in range(batch_size):
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

# 
# redundant function to emit per-frame predictions
# 
def test_per_frame(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device="cpu"):
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

            # 
            # instead onf classifying the whole video, classify frames
            # 
            batch_outputs = model(videos)

            # last cell again
            loss = model.criterion(batch_outputs[:, -1, :], targets) * TEST_LOSS_MULTIPLY
            total_loss += loss.item()
            running_loss += loss.item()

            # get rid of batch
            batch_outputs.squeeze_(0)
            
            # for each frame in video
            # for frame_outputs in batch_outputs:
            for i in range(batch_outputs.shape[0]):
                # pick index with max activation value
                _, predicted_indices = torch.max(batch_outputs[i,:].data, 0)
                predicted_indices.unsqueeze_(0)
                
                # bin predictions into confusion matrix
                for j in range(len(videos)):
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
    train(model, loader_train, loader_test, n_classes, epochs=75, save=False, masked=True, device=device)


