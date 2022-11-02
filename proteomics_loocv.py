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
def set_datasets(csv1,csv2,ind): #two csv files for proteomics. Ratio between 0 to 10.
    # processing protein spreadsheets.
    protFrame1 = pd.read_csv(csv1)
    protFrame2 = pd.read_csv(csv2)
    protFrame2 = protFrame2.iloc[10:, :]
    train_protFrame = pd.concat([protFrame1, protFrame2])
    train_protFrame = train_protFrame.values.tolist()


    # Changing the labels from healthy,Long covid, and No long COVID to 0 or 1 (0 for no long covid and healthy, 1 for yes long covid)
    for i in range(len(train_protFrame)):
        if train_protFrame[i][5] == "Long covid":
            train_protFrame[i][5] = 1;
        else:
            train_protFrame[i][5] = 0;

    # Making duplicate for the testing set
    test_protFrame = train_protFrame[:]

    source_indices = np.arange(len(train_protFrame))
    keep_indices = np.delete(source_indices, [ind])
    for index in sorted (keep_indices,reverse=True):
        del test_protFrame[index]
    del train_protFrame[ind]


    # Send to pytorch dataset class
    train_set = ProtDataset(train_protFrame)
    test_set = ProtDataset(test_protFrame)

    return train_set,test_set

## Experiment regime: LOOCV with batch size variance.
epochs = [500]
lrs = [0.001]
batch_size = [32]
datasize = 72
record = np.zeros((datasize,len(batch_size)))
trainsets = []
testsets = []
trainloaders = []
testloaders = []

for ind in range(datasize):
    trainset, testset = set_datasets('data/4-4-22_Test.AllLC_vsHealthy_Data.csv',
                                           'data/4-4-22_Test.AllNLC_vsHealthy_Data.csv',ind)
    trainsets.append(trainset)
    testsets.append(testset)
    trainloaderpre = []
    testloaderpre = []
    for j in range(len(batch_size)):
        trainloader = torch.utils.data.DataLoader(trainsets[ind], batch_size=batch_size[j], shuffle=True)
        testloader = torch.utils.data.DataLoader(testsets[ind], batch_size=1, shuffle=False)
        trainloaderpre.append(trainloader)
        testloaderpre.append(testloader)
    trainloaders.append(trainloaderpre)
    testloaders.append(testloaderpre)

##
for q in range(datasize):
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
    PATH = './models/batch_adjust_looc/mlp1_proteomics_reset_' + str(q) + '.pth'
    torch.save(mlp1.state_dict(), PATH)

    ##
    criterion1 = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(mlp1.parameters(), lr=0.001, momentum=0.9)

    for j in range(len(batch_size)):
        n_epoch = 500
        t1_start = perf_counter()
        for epoch in range(n_epoch):  # loop over the dataset multiple times
          for i, data in enumerate(trainloaders[q][j], 0):
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

        PATH2 = './models/batch_adjust_looc/mlp1_proteomics_' + str(q) + str(j) + '.pth'
        torch.save(mlp1.state_dict(), PATH2)
        mlp1.load_state_dict(torch.load(PATH2))

        correct = 0

        with torch.no_grad():
            for data in testloaders[q][j]:
                inputs1 = data['profile']
                labels1 = data['label']
                labels1 = labels1.to(device)
                outputs1 = mlp1(inputs1)
                predicted1 = torch.max(outputs1.data)
                if predicted1 == labels1:
                    correct = 1
                pass
        record[q,j] = correct
    print(record)
print(record)

## to do: actually debug and see if the prediction/accuracy code is correct. 
