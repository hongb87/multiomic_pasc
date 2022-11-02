##
import torch
import math
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from torch.utils.data import Dataset
import pandas as pd
import random
from random import sample

## Sets the device to the GPU (mps is for apple m1 chip)
device = torch.device("mps" if torch.has_mps else "cpu")
print(device)

## Dataset and loader.
class ProtDataset(Dataset):
    def __init__(self,prot_frame,transform=None):
        self.protFrame = prot_frame
        self.transform = transform

    def __len__(self):
        return len(self.protFrame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pat_id = self.protFrame[idx][2]
        pat_label = self.protFrame[idx][5]
        pat_profile = self.protFrame[idx][7:]
        pat_profile = np.asarray(pat_profile)
        pat_profile = pat_profile.astype('float')
        sample = {'id': pat_id,'label':pat_label,'profile': pat_profile}

        if self.transform:
            sample = self.transform(sample)

        return sample

## Funtion for creating the testing and training datasets. Combining 2 csv files for proteomics. Must also randomize the training and testing sets. The random
# indices must be generated outside the class because we want to keep track of which elements we remove
def set_datasets(csv1,csv2,training_ratio): #two csv files for proteomics. Ratio between 0 to 10.
    # processing protein spreadsheets.
    protFrame1 = pd.read_csv(csv1)
    protFrame2 = pd.read_csv(csv2)
    protFrame2 = protFrame2.iloc[10:, :]
    train_protFrame = pd.concat([protFrame1, protFrame2])
    train_protFrame = train_protFrame.values.tolist()
    print(len(train_protFrame))
    print(train_protFrame[0][5])

    # Changing the labels from healthy,Long covid, and No long COVID to 0 or 1 (0 for no long covid and healthy, 1 for yes long covid)
    for i in range(len(train_protFrame)):
        if train_protFrame[i][5] == "Long covid":
            train_protFrame[i][5] = 1;
        else:
            train_protFrame[i][5] = 0;

    # Making duplicate for the testing set
    test_protFrame = train_protFrame[:]

    # randomizing
    total_length = len(train_protFrame)
    to_drop = int(round(((1-training_ratio) * total_length)))
    source_indices = np.arange(total_length)
    drop_indices = np.random.choice(source_indices, to_drop, replace=False)
    keep_indices = np.delete(source_indices, drop_indices)
    for index in sorted(drop_indices, reverse=True):
        del train_protFrame[index]

    for index in sorted (keep_indices,reverse=True):
        del test_protFrame[index]

    # Send to pytorch dataset class
    train_set = ProtDataset(train_protFrame)
    test_set = ProtDataset(test_protFrame)

    return train_set,test_set

## Experiment regime: 200,400,600,800,1000 epochs, full batch size (using full batch size because the dataset is small anyways). Learning rate: 0.01,0.001,0.0001,0.00001
epochs = [200,400,600,800,1000]
lrs = [0.01,0.001,0.0001,0.00001]
num_models = 4
record = np.zeros((len(epochs),len(lrs),num_models))

training_ratio = 0.8
train_protein, test_protein = set_datasets('data/4-4-22_Test.AllLC_vsHealthy_Data.csv',
                                           'data/4-4-22_Test.AllNLC_vsHealthy_Data.csv', training_ratio)
train_loader = torch.utils.data.DataLoader(train_protein, batch_size=len(train_protein), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_protein, batch_size=1, shuffle=False)

##
count1 = 0
for q in epochs:
    count2 = 0
    for j in lrs:
        # Original net
        class MLP(nn.Module):
          def __init__(self):
            super(MLP, self).__init__()
            # TODO: define your MLP
            self.m1 = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(7596, 2048)
            self.m2 = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(2048, 1024)
            self.m3 = nn.Dropout(p=0.3)
            self.fc3 = nn.Linear(1024, 512)
            self.fc4 = nn.Linear(512, 1)

          def forward(self, x):
            # TODO: define your forward function
            x = x.type(torch.FloatTensor)
            x = x.to(device)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.m2(x)
            x = F.relu(self.fc2(x))
            x = self.m3(x)
            x = F.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x

        mlp = MLP().to(device)
        print(mlp)
        PATH = './models/NNadjust/mlp_proteomics_reset_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp.state_dict(), PATH)

        # Added one layer, 7596 to 4096
        class MLP1(nn.Module):
          def __init__(self):
            super(MLP1, self).__init__()
            # TODO: define your MLP
            self.m1 = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(7596, 4096)
            self.m2 = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(4096,2048)
            self.m3 = nn.Dropout(p=0.3)
            self.fc3 = nn.Linear(2048, 1024)
            self.m4 = nn.Dropout(p=0.3)
            self.fc4 = nn.Linear(1024, 512)
            self.fc5 = nn.Linear(512, 1)

          def forward(self, x):
            # TODO: define your forward function
            x = x.type(torch.FloatTensor)
            x = x.to(device)
            x = torch.flatten(x, 1)
            x = self.m1(x)
            x = F.relu(self.fc1(x))
            x = self.m2(x)
            x = F.relu(self.fc2(x))
            x = self.m3(x)
            x = F.relu(self.fc3(x))
            x = self.m4(x)
            x = F.relu(self.fc4(x))
            x = torch.sigmoid(self.fc5(x))
            return x

        mlp1 = MLP1().to(device)
        print(mlp1)
        PATH = './models/NNadjust/mlp1_proteomics_reset_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp1.state_dict(), PATH)

        # Added two layers, 7596 to 4096 and 512 to 256
        class MLP2(nn.Module):
          def __init__(self):
            super(MLP2, self).__init__()
            # TODO: define your MLP
            self.m1 = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(7596, 4096)
            self.m2 = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(4096,2048)
            self.m3 = nn.Dropout(p=0.3)
            self.fc3 = nn.Linear(2048, 1024)
            self.m4 = nn.Dropout(p=0.3)
            self.fc4 = nn.Linear(1024, 512)
            self.m5 = nn.Dropout(p=0.3)
            self.fc5 = nn.Linear(512, 256)
            self.fc6 = nn.Linear(256,1)

          def forward(self, x):
            # TODO: define your forward function
            x = x.type(torch.FloatTensor)
            x = x.to(device)
            x = torch.flatten(x, 1)
            x = self.m1(x)
            x = F.relu(self.fc1(x))
            x = self.m2(x)
            x = F.relu(self.fc2(x))
            x = self.m3(x)
            x = F.relu(self.fc3(x))
            x = self.m4(x)
            x = F.relu(self.fc4(x))
            x = self.m5(x)
            x = F.relu(self.fc5(x))
            x = torch.sigmoid(self.fc6(x))
            return x

        mlp2 = MLP2().to(device)
        print(mlp2)
        PATH = './models/NNadjust/mlp2_proteomics_reset_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp2.state_dict(), PATH)

        # Added four layers, 7596 to 4096 and 512 to 256,256 to 128, 128 to 64
        class MLP3(nn.Module):
          def __init__(self):
            super(MLP3, self).__init__()
            # TODO: define your MLP
            self.m1 = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(7596, 4096)
            self.m2 = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(4096,2048)
            self.m3 = nn.Dropout(p=0.3)
            self.fc3 = nn.Linear(2048, 1024)
            self.m4 = nn.Dropout(p=0.3)
            self.fc4 = nn.Linear(1024, 512)
            self.m5 = nn.Dropout(p=0.3)
            self.fc5 = nn.Linear(512, 256)
            self.m6 = nn.Dropout(p=0.3)
            self.fc6 = nn.Linear(256,128)
            self.m7 = nn.Dropout(p=0.3)
            self.fc7 = nn.Linear(128,64)
            self.fc8 = nn.Linear(64,1)


          def forward(self, x):
            # TODO: define your forward function
            x = x.type(torch.FloatTensor)
            x = x.to(device)
            x = torch.flatten(x, 1)
            x = self.m1(x)
            x = F.relu(self.fc1(x))
            x = self.m2(x)
            x = F.relu(self.fc2(x))
            x = self.m3(x)
            x = F.relu(self.fc3(x))
            x = self.m4(x)
            x = F.relu(self.fc4(x))
            x = self.m5(x)
            x = F.relu(self.fc5(x))
            x = self.m6(x)
            x = F.relu(self.fc6(x))
            x = self.m7(x)
            x = F.relu(self.fc7(x))
            x = torch.sigmoid(self.fc8(x))
            return x

        mlp3 = MLP3().to(device)
        print(mlp3)
        PATH = './models/NNadjust/mlp3_proteomics_reset_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp3.state_dict(), PATH)

        ##
        criterion = nn.CrossEntropyLoss()
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        criterion3 = nn.CrossEntropyLoss()

        optimizer = optim.SGD(mlp.parameters(), lr=j, momentum=0.9)
        optimizer1 = optim.SGD(mlp1.parameters(), lr=j, momentum=0.9)
        optimizer2 = optim.SGD(mlp2.parameters(), lr=j, momentum=0.9)
        optimizer3 = optim.SGD(mlp3.parameters(), lr=j, momentum=0.9)

        ## training original model
        n_epoch = q

        t1_start = perf_counter()
        for epoch in range(n_epoch):  # loop over the dataset multiple times
          for i, data in enumerate(train_loader, 0):
            # TODO: write training code

            inputs = data['profile']
            labels = data['label']
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                                t1_stop-t1_start)

        PATH1 = './models/NNadjust/mlp_proteomics_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp.state_dict(), PATH1)
        mlp.load_state_dict(torch.load(PATH1))

        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                inputs = data['profile']
                labels = data['label']
                labels = labels.to(device)
                outputs = mlp(inputs)
                predicted = torch.max(outputs.data)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                pass

        print(f'Accuracy: {100 * correct // total} %')
        record[count1,count2,0] = 100 * correct // total

        ## training model 1
        n_epoch = q

        t1_start = perf_counter()
        for epoch in range(n_epoch):  # loop over the dataset multiple times
          for i, data in enumerate(train_loader, 0):
            # TODO: write training code

            inputs1 = data['profile']
            labels1 = data['label']
            labels1 = labels1.to(device)
            optimizer1.zero_grad()
            outputs1 = mlp1(inputs1)
            loss1 = criterion1(outputs1, labels1)
            loss1.backward()
            optimizer1.step()
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                                t1_stop-t1_start)

        PATH2 = './models/NNadjust/mlp1_proteomics_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp1.state_dict(), PATH2)
        mlp1.load_state_dict(torch.load(PATH2))

        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                inputs1 = data['profile']
                labels1 = data['label']
                labels1 = labels1.to(device)
                outputs1 = mlp1(inputs1)
                predicted1 = torch.max(outputs1.data)
                total = total + labels1.size(0)
                correct = correct + (predicted1 == labels1).sum().item()
                pass

        print(f'Accuracy: {100 * correct // total} %')
        record[count1,count2,1] = 100 * correct // total

        ## training model 2
        n_epoch = q

        t1_start = perf_counter()
        for epoch in range(n_epoch):  # loop over the dataset multiple times
          for i, data in enumerate(train_loader, 0):
            # TODO: write training code

            inputs2 = data['profile']
            labels2 = data['label']
            labels2 = labels2.to(device)
            optimizer2.zero_grad()
            outputs2 = mlp2(inputs2)
            loss2 = criterion2(outputs2, labels2)
            loss2.backward()
            optimizer2.step()
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                                t1_stop-t1_start)

        PATH3 = './models/NNadjust/mlp2_proteomics_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp2.state_dict(), PATH3)
        mlp2.load_state_dict(torch.load(PATH3))

        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                inputs2 = data['profile']
                labels2 = data['label']
                labels2 = labels2.to(device)
                outputs2 = mlp2(inputs2)
                predicted2 = torch.max(outputs2.data)
                total = total + labels2.size(0)
                correct = correct + (predicted2 == labels2).sum().item()
                pass

        print(f'Accuracy: {100 * correct // total} %')
        record[count1,count2,2] = 100 * correct // total

        ## training model 3
        n_epoch = q

        t1_start = perf_counter()
        for epoch in range(n_epoch):  # loop over the dataset multiple times
          for i, data in enumerate(train_loader, 0):
            # TODO: write training code

            inputs3 = data['profile']
            labels3 = data['label']
            labels3 = labels3.to(device)
            optimizer3.zero_grad()
            outputs3 = mlp3(inputs3)
            loss3 = criterion(outputs3, labels3)
            loss3.backward()
            optimizer3.step()
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                                t1_stop-t1_start)

        PATH4 = './models/NNadjust/mlp3_proteomics_' + str(count1) + str(count2) + '.pth'
        torch.save(mlp3.state_dict(), PATH4)
        mlp3.load_state_dict(torch.load(PATH4))

        correct = 0
        total = 0

        t1_start = perf_counter()
        with torch.no_grad():
            for data in test_loader:
                inputs3 = data['profile']
                labels3 = data['label']
                labels3 = labels3.to(device)
                outputs3 = mlp3(inputs3)
                predicted3 = torch.max(outputs3.data)
                total = total + labels3.size(0)
                correct = correct + (predicted == labels3).sum().item()
                pass
        t1_stop = perf_counter()

        print(f'Accuracy: {100 * correct // total} %')
        record[count1,count2,3] = 100 * correct // total

        ##
        print(record)
        count2 = count2 + 1
    count1 = count1 + 1
##
print(record)

