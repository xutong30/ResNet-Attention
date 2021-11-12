import pandas as pd
import Data_prepro
import Models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

# read csv files that include all training data and validation data
train_d = pd.read_csv('train.csv')
valid_d = pd.read_csv('vali.csv')

# label species to numbers
label_list = {"Aegypti": 0, "albopictus": 1, "arabiensis": 2, "gambiae" : 3, "quinquefasciatus" : 4, "pipiens" : 5}
no_class=6

# load training data
print('loading training data...')
data_train = './train_pad'
label_train = './train.csv'
train_data = Data_prepro.ListDataset(data_train, label_train, label_list)

# load validation data
print('loading validation data...')
data_vali = './vali_pad'
label_vali = './vali.csv'
valid_data = Data_prepro.ListDataset(data_vali, label_vali, label_list)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

# arrange GPU and load our model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_model = Models.resnet18(1, 6) # 1 channel, 6 classes
resnet_model = resnet_model.to(device)


# start training the model and evaluating the model, and save the related measurements into a text file.
def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, train_acc, valid_acc, sched):
    output_file = open('wing18_aug.txt', 'w+')
    output_file.write('start')
    output_file.write('\n')
    output_file.close()
    print('start')
    # training procedure
    for epoch in range(1, epochs+1):
        start_time = time.time()
        model.train()
        batch_losses=[]
        train_y = []
        train_yhat = []
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            train_y.append(y.cpu().detach().numpy())
            train_yhat.append(y_hat.cpu().detach().numpy())
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        train_y = np.concatenate(train_y)
        train_yhat = np.concatenate(train_yhat)
        accuracy = np.mean(train_yhat.argmax(axis=1)==train_y)
        train_acc.append(accuracy)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])} Train-Accuracy : {accuracy}')
        output_file = open('wing18_aug.txt', 'a')
        output_file.write(
            f"Epoch {epoch}/{epochs}.. Train loss: {np.mean(train_losses[-1])}.. ")
        output_file.write('\n')
        output_file.write(
            f"Epoch {epoch}/{epochs}.. Train acc: {accuracy}.. ")
        output_file.write('\n')
        output_file.close()
        sched.step()

        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        # validating procedure
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)

        y_test = trace_y
        y_pred = trace_yhat.argmax(axis=1)

        # calculating measurements

        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        valid_acc.append(accuracy)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')
        # precision, recall, f1-score
        output_file = open('wing18_aug.txt', 'a')
        output_file.write('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
        output_file.write('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
        output_file.write('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
        output_file.write('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
        output_file.write('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
        output_file.write('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
        output_file.write('\nClassification Report\n')
        output_file.write(classification_report(y_test, y_pred))
        output_file.write('\n')

        # confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        output_file.write(str(conf_matrix))
        output_file.write('\n')

        # accuracy and loss
        output_file.write(
            f"Epoch {epoch}/{epochs}.. Vali loss: {np.mean(valid_losses[-1])}.. ")
        output_file.write('\n')
        output_file.write(
            f"Epoch {epoch}/{epochs}.. Vali acc: {accuracy}.. ")
        output_file.write('\n')

        # time usage for each epoch
        end_time = time.time()
        usage_time = end_time - start_time
        output_file.write(f"Time: {usage_time}..")
        output_file.write('\n')

        output_file.close()

# set learning rate and optimizer
learning_rate = 2e-4
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)

# set learning rate schedule
sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 0)

# set epoch numbers and loss function
epochs = 100
loss_fn = nn.CrossEntropyLoss()

# set measurement arrays
resnet_train_losses = []
resnet_valid_losses = []
resnet_train_acc = []
resnet_valid_acc = []

train(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses, resnet_valid_losses, resnet_train_acc, resnet_valid_acc,sched)

