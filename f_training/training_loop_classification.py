# imports
import torch

import time
import datetime
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    '''
    trains the model, and returns the model with best accuracy and history 
    for loss and accuracy in both training and validation set
    '''
    
    modelname = model.__class__.__name__
    start = datetime.now()

    loss_history = {'train': [], 'val': []}
    acc_history = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f'Training model {modelname}')
    print(start.strftime("%d/%m/%Y %H:%M:%S"), '\n')
    print('*'*110, '\n')

    for epoch in range(1, num_epochs+1):
        
        epoch_loss_dict = dict()
        epoch_acc_dict = dict()

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            # metrics 
            running_loss = 0.0
            running_corrects = 0

            # iterate over batches
            for inputs, labels in dataloaders[phase]:

                # get data to the right device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track gradient history only if training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # if training: backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # record epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_loss_dict[phase] = epoch_loss
            epoch_acc_dict[phase] = epoch_acc

            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

            # check for best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # end of epoch
        nowtime = datetime.now().strftime("%H:%M:%S")
        if epoch < 4 or epoch % 10 == 0:
            msg_loss = f"train_loss: {epoch_loss_dict['train']:.4f}  valid_loss: {epoch_loss_dict['val']:.4f}"
            msg_acc = f"train_acc: {epoch_acc_dict['train']:.4f}  valid_acc: {epoch_acc_dict['val']:.4f}"
            print(f'epoch {epoch}/{num_epochs}\t{msg_loss}  {msg_acc}   {nowtime}')

    # end of training
    time_elapsed = datetime.now() - start
    time_elapsed_msg = str(time_elapsed).split('.')[0]
    print()
    print('*'*110)
    print(f'\n{modelname} training completed in ', time_elapsed_msg)
    print(f'Best validation Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    log = {'loss': loss_history, 'acc': acc_history}

    return model, log