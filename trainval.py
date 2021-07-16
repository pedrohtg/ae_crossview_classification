from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from tqdm import tqdm
from collections import OrderedDict

def calculate_metrics(preds, labels, file = None):
    cm = confusion_matrix(np.asarray(labels), np.asarray(preds))
    b_acc = balanced_accuracy_score(np.asarray(labels), np.asarray(preds))
    acc = accuracy_score(np.asarray(labels), np.asarray(preds))
    kappa = cohen_kappa_score(np.asarray(labels), np.asarray(preds))
    f1 = f1_score(np.asarray(labels), np.asarray(preds), average = 'weighted')	
    if file is not None:
        file.write("Accuracy: " + str(acc) + "\n")
        file.write("Balanced_Accuracy: " + str(b_acc) + "\n")
        file.write("Kappa: " + str(kappa) + "\n")
        file.write("F1: " + str(f1) + "\n")
        file.write("Confusion Matrix: \n" + str(cm) + "\n\n\n\n")
        file.write("Labels\n{}\n".format(np.asarray(labels)))
        file.write("Predictions\n{}".format(np.asarray(preds)))
        
    else:
    	print ("\nAccuracy: " + str(acc))
    	print ("Balanced_Accuracy: " + str(b_acc))
    	print ("Kappa: " + str(kappa))
    	print ("F1: " + str(f1))
    	print (cm)

def train(model, dataloaders, criterion, optimizer, num_epochs, epochs_early_stop, 
          tensor_board, is_vae,
          alpha_1=1, alpha_2=1):
    counter_early_stop_epochs = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    total_time = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 9999999.99

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        predictions = []
        labels_list = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data_a, data_g in tqdm(dataloaders[phase]):
                inp_a = data_a[0][0].to(device)
                inp_g = data_g[0][0].to(device)
                labels = data_a[0][1].to(device)
                # inputs = inputs[0].to(device)
                # labels = labels[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if (phase == 'train'):
                        time1 = time.time()

                    if is_vae and phase == 'train':
                        rec_a, rec_g, clf, *aux_outputs = model(inp_a, inp_g)
                        rec_loss = criterion[0]((rec_a, rec_g), (inp_a, inp_g), *aux_outputs)
                        clf_loss = criterion[1](clf, labels)
                        loss = alpha_1*rec_loss + alpha_2*clf_loss
                    else:
                        # Get model outputs and calculate loss
                        rec_a, rec_g, clf = model(inp_a, inp_g)
                        rec_loss = criterion[0](rec_a, inp_a) +  criterion[0](rec_g, inp_g)
                        clf_loss = criterion[1](clf, labels)
                        loss = alpha_1*rec_loss + alpha_2*clf_loss
                    
                    _, preds = torch.max(clf, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if (phase == 'train'):
                        total_time += (time.time() - time1)

                # statistics
                for p in preds.data.cpu().numpy(): 
                    predictions.append(p)
                for l in labels.data.cpu().numpy(): 
                    labels_list.append(l)
                running_loss += loss.item() * inp_a.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if (phase == 'train'):
                tensor_board.add_scalar('Loss/train', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                tensor_board.add_scalar('Loss/val', epoch_loss, epoch)
                tensor_board.add_scalar('Accuracy/val', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val':
               counter_early_stop_epochs += 1
               val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_loss < best_val_loss:
                counter_early_stop_epochs = 0
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            calculate_metrics(predictions, labels_list)
            
            predictions = []
            labels_list = []
        print ('Epoch ' + str(epoch) + ' - Time Spent ' + str(total_time))
        if (counter_early_stop_epochs >= epochs_early_stop):
            print ('Stopping training because validation loss did not improve in ' + str(epochs_early_stop) + ' consecutive epochs.')
            break
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def final_eval(model, dataloaders, csv_file, stats_file, is_vae):

    def softmax(A):
        e = np.exp(A)
        return  e / e.sum(axis=0).reshape((-1,1))

    print ("Begining final eval.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_file.write('Image;Labels;Predictions;Softmax\n')
    predictions = []
    labels_list = []
    softmax_values = []
    image_names = []

    for data_a, data_g in dataloaders['val']:
        # Saving name of images
        
        for names in data_a[1][0]: 
            image_names.append(names)
        
        inp_a = data_a[0][0].to(device)
        inp_g = data_g[0][0].to(device)
        labels = data_a[0][1].to(device)
        
        if is_vae:
            rec_a, rec_g, clf, *_ = model(inp_a, inp_g)
        else:
            rec_a, rec_g, clf = model(inp_a, inp_g)
        
        _, preds = torch.max(clf, 1)

        for p in preds.data.cpu().numpy(): 
            predictions.append(p)
        for l in labels.data.cpu().numpy(): 
            labels_list.append(l)
        for s in range(len(preds)):
            for o in softmax(clf.data.cpu().numpy()[s]):
                softmax_values.append(o)

    for i in range(len(predictions)):
        csv_file.write(str(image_names[i]) + ';' + str(labels_list[i]) + ';' + str(predictions[i]) + ';' + str(softmax_values[i]) + '\n')
    calculate_metrics(predictions, labels_list, stats_file)