import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_losses = []
testing_losses = []

def train(model, train_loader, test_loader, n_classes, epochs=10):
    minibatch_count = len(train_loader)
    print('training minibatch count:', minibatch_count)

    for epoch in range(epochs):
        #  rnn
        # hidden = model.initHidden(BATCH_SIZE)
        # hidden = hidden.to(device)

        total_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            model.optimizer.zero_grad()

            outputs = model(inputs)

            # rnn
            # hidden = hidden[:inputs.shape[0]]
            # outputs, hidden = model(inputs, hidden)

            loss = model.criterion(outputs, targets)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            running_loss += loss.item()

            if i % 5 == 0:
                print('epoch: %d minibatches: %d/%d loss: %.3f' % (epoch+1, i+1, minibatch_count, running_loss))
                running_loss = 0.0
        
        training_losses.append(total_loss)
        test(model, test_loader, n_classes)

    print('Finished Training')
    print('Training losses:', training_losses)
    print('Testing losses:', testing_losses)

    # save model
    torch.save(model.state_dict(), "saved_models/simple_uint8_masked.pt")

def test(model, test_loader, n_classes):
    correct = 0
    total = 0
    class_correct = [0]*n_classes
    class_total = [0]*n_classes

    running_loss = 0.0
    total_loss = 0.0
    confusion = torch.zeros([n_classes, n_classes], dtype=torch.int) # (class, guess)

    minibatch_count = len(test_loader)
    print('testing minibatch count:', minibatch_count)

    # rnn
    # hidden = model.initHidden(BATCH_SIZE)
    # hidden = hidden.to(device)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            # rnn
            # hidden = hidden[:images.shape[0]]
            # outputs, hidden = model(images, hidden)

            outputs = model(images)

            loss = model.criterion(outputs, targets)
            total_loss += loss.item()
            running_loss += loss.item()
            
            _, predicted_indexes = torch.max(outputs.data, 1)
            
            # bin predictions into confusion matrix
            for j in range(len(images)):
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
                print('minibatches: %d/%d running loss: %.3f' % (i+1, minibatch_count, running_loss))
                running_loss = 0.0
        
        testing_losses.append(total_loss)

    print('Correct predictions:', class_correct)
    print('Total test samples: ', class_total)
    print('Test accuracy: %d %%' % (100 * correct / total))
    print(confusion)
