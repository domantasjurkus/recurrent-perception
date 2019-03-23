import numpy as np
import torch
import torch.nn.functional as F

training_losses = []
testing_losses = []
accuracies = []

STRIDE = 3

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

            # slide window with STRIDE
            for i in range(fps, timesteps+1, STRIDE):
                feature_sequence = videos[:, i-fps:i, :]
                outputs, hc = model(feature_sequence, hc)
                hc = (hc[0].detach(), hc[1].detach())

                # backpropagate for each snippet
                loss = model.criterion(outputs, targets)
                # loss.backward(retain_graph=True)
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

            batch, timesteps, _, _, _ = videos.size()

            hc = init_hidden_and_cell(device=device)
            prediction_bins = torch.zeros([1, n_classes], device=device)

            # slide window with STRIDE
            for i in range(fps, timesteps+1, STRIDE):
                feature_sequence = videos[:, i-fps:i, :]
                outputs, hc = model(feature_sequence, hc)

                softmax_predictions = F.softmax(outputs, dim=1)
                prediction_bins += softmax_predictions

                # backpropagate for each snippet
                loss = model.criterion(outputs, targets) * TEST_LOSS_MULTIPLY

                total_loss += loss.item()
                running_loss += loss.item()

            #TODO: extend with accuracy metric (correctly classified snippets / number of snippets)
            #TODO: the same for single shot, but correctly classified frames/ number of frames

            # pick index with max activation value
            _, predicted_indexes = torch.max(prediction_bins.data, 1)
            
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