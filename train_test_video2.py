import torch
import numpy as np
from statistics import mode

training_losses = []
testing_losses = []
accuracies = []

# unused I think

def train(model, train_loader, test_loader, n_classes, epochs=10, masked=False, save=False, device="cpu", fps=6):
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
            model.optimizer.zero_grad()

            outputs = get_outputs(model, video, targets, n_classes, fps, device)

            targets = targets.repeat(outputs.shape[0])
            targets.to(device)

            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            if i % 5 == 0:
                print('epoch: %d minibatches: %d/%d loss: %.3f' % (epoch, i, minibatch_count, running_loss))
                running_loss = 0.0
        
        training_losses.append(total_loss)
        acc = test(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device=device, fps=fps)
        print('Total training loss:', total_loss)

        if save:
            save_model(model, masked, epoch, acc)

        print('Training losses:', str(training_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Testing losses:', str(testing_losses).replace(",", "").replace("[", "").replace("]", ""))
        print('Accuracies:', str(accuracies).replace(",", "").replace("[", "").replace("]", ""))

    print('Finished Training')

def get_outputs(model, inputs, targets, n_classes, fps, device):
    # split video, feed each snippet
    n_snippets = inputs.shape[1] // fps

    # hard-coded batch size of 1
    # The output for each snippet:
    snippet_outputs = torch.zeros(n_snippets, n_classes, device=device)

    # unstrided
    for i in range(0,n_snippets):
        start = i*fps
        end = (i+1)*fps
        classes = model(inputs[:,start:end,:,:,:])
        snippet_outputs[i] = classes
    
    return snippet_outputs

def save_model(model, masked, epoch, acc):
    model_name = type(model).__name__.lower()
    is_masked = 'masked' if masked else 'depth'
    torch.save(model.state_dict(), "saved_models/%s_%s_epoch%d_acc%f.pt" % (model_name, is_masked, epoch, acc))
    print("model saved")

def test(model, test_loader, n_classes, TEST_LOSS_MULTIPLY, device="cpu", fps=6):
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
            videos, targets, video_lengths = data
            videos, targets = videos.to(device, dtype=torch.float), targets.to(device)

            outputs = get_outputs(model, videos, targets, n_classes, fps, device)

            targets = targets.repeat(outputs.shape[0])
            targets.to(device)

            # outputs.shape = (1, n_snippets, n_classes)
            loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY
            total_loss += loss.item()
            running_loss += loss.item()

            targets = torch.tensor([targets[0]])
            
            # pick prediction for each snippet that has the highest activation
            _, predicted_snippet_indexes = torch.max(outputs.data, 1)

            # Use torch.mode() to find which class has most snippets predicted for the video.
            # If 2 classes have the most snippets, the lower index wins (bias towards pants).
            # print(predicted_snippet_indexes)
            predicted_video_index, _ = predicted_snippet_indexes.mode(dim=0)
            # print(predicted_video_index)
            
            # bin predictions into confusion matrix
            for j in range(len(videos)):
                actual = targets[j].item()
                # predicted = predicted_video_index[j].item()
                predicted = predicted_video_index.item()
                confusion[actual][predicted] += 1

            # print(targets)

            # sum probabilities at test time


            # sum up total correct
            # batch_size = targets.size(0)
            batch_size = 1
            total += batch_size
            correct_vector = (predicted_video_index == targets.item())
            correct += correct_vector.sum().item()

            # sum up per-class correct - if batch_size > 1:
            # for j in range(len(targets)):
            #     target = targets[j]
            #     class_correct[target] += correct_vector.item()
            #     class_total[target] += 1

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