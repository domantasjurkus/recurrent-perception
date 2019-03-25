import torch
import numpy as np

from datasets.xtion1video import Xtion1VideoDataset
from models.snippet_concat import SnippetConcat

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

BATCH_SIZE = 1
SHUFFLE = True
FRAMES_PER_SEQUENCE = 6
STRIDE = 6

classes = ('pant', 'shirt', 'sweater', 'towel', 'tshirt')
n_classes = len(classes)

# dataset_train = Xtion1SnippetDataset(root='../project-data/continuous_masked', frames_per_sequence=FRAMES_PER_SEQUENCE)
# dataset_test = Xtion1SnippetDataset(root='../project-data/continuous_masked_test', frames_per_sequence=FRAMES_PER_SEQUENCE)

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
    model = SnippetConcat(n_classes=n_classes)
    return model

model = get_model()
model.to(device)

training_losses = []
testing_losses = []
accuracies = []

def train(model, train_loader, test_loader, n_classes, epochs=10, masked=True, save=False, device="cpu", fps=6):
    minibatch_count = len(train_loader)
    print('training minibatch count:', minibatch_count)
    print("device:", device)

    TEST_LOSS_MULTIPLY = len(train_loader)/len(test_loader)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            video, targets, _ = data
            video, targets = video.to(device, dtype=torch.float), targets.to(device)
            
            batchsize, timesteps, _, _, _ = video.shape

            for end in range(fps, timesteps, STRIDE):
                start = end-fps
                model.optimizer.zero_grad()
            
                outputs = model(video[:, start:end, :])

                loss = model.criterion(outputs, targets)
                loss.backward()
                model.optimizer.step()

                total_loss += loss.item()
                running_loss += loss.item()

            if i % 5 == 0:
                print('epoch: %d minibatches: %d/%d loss: %.3f' % (epoch, i, minibatch_count, running_loss))
                running_loss = 0.0
        
        training_losses.append(total_loss)
        acc = test_video(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device=device, fps=fps)
        print('Total training loss:', total_loss)
        print('Training losses:', str(training_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Testing losses:', str(testing_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Accuracies:', str(accuracies).replace(",", "").replace("[", "").replace("]", ""))

    print('Finished Training')

def test_video(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device="cpu", fps=6):
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
            video, targets, video_lengths = data
            video, targets = video.to(device, dtype=torch.float), targets.to(device)
            batchsize, timesteps, _, _, _ = video.shape

            snippet_class_bins = [0]*n_classes

            for end in range(fps, timesteps, STRIDE):
                start = end-fps
            
                outputs = model(video[:, start:end, :])
                loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY

                total_loss += loss.item()
                running_loss += loss.item()

                _, predicted_snippet_index = torch.max(outputs.data, 1)
                predicted_snippet_index = predicted_snippet_index.item()
                snippet_class_bins[predicted_snippet_index] += 1

            predicted_video_index = np.argmax(snippet_class_bins)
            
            # bin prediction into confusion matrix
            actual = targets.item()
            predicted = predicted_video_index
            confusion[actual][predicted] += 1

            # sum up total correct
            # batch_size = targets.size(0)
            batch_size = 1
            total += batch_size
            correct_vector = (predicted_video_index == targets.item())
            correct += correct_vector.sum().item()

            # if batch_size = 1:
            class_correct[targets[0]] += correct_vector.item()
            class_total[targets[0]] += 1

            if i % 5 == 0:
                print('minibatches: %d/%d running loss: %.3f' % (i, minibatch_count, running_loss))
                running_loss = 0.0
        
        testing_losses.append(total_loss)

    print('Total test samples: ', class_total)
    print('Correct predictions:', class_correct)
    acc = correct / total
    accuracies.append(acc)
    print('Test accuracy: %f' % acc)
    print(confusion)
    print('Total testing loss:', total_loss)
    return acc

if __name__ == '__main__':  
    train(model, loader_train, loader_test, n_classes, epochs=50, save=False, masked=True, device=device)
