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

## Example of csv read, sample 20
protFrame1 = pd.read_csv('data/4-4-22_Test.AllLC_vsHealthy_Data.csv')
protFrame2 = pd.read_csv('data/4-4-22_Test.AllNLC_vsHealthy_Data.csv')
protFrame2 = protFrame2.iloc[10:,:]


protFrame = pd.concat([protFrame1,protFrame2])
protFrame = protFrame.values.tolist()

n = 19
pat_id2 = protFrame[n][2]
pat_label2 = protFrame[n][5]
pat_profile2 = protFrame[n][7:]
pat_profile2 = np.asarray(pat_profile2)

print(np.shape(protFrame))
print(protFrame[1][:5])

print('pat_id: {}'.format(pat_id2))
print('pat_label: {}'.format(pat_label2))
print('pat_profile: {}'.format(pat_profile2.shape))
print('First 4 Landmarks: {}'.format(pat_profile2[:4]))
print(len(protFrame))

indexes = np.arange(len(protFrame))
indices = np.random.choice(indexes,3,replace=False)
n_indexes = np.delete(indexes,indices)
print(n_indexes)
print(len(n_indexes))
print(indices)
indices=[]
for index in sorted(indices, reverse=True):
    del protFrame[index]
print(np.shape(protFrame))
print(protFrame[1][:5])

n_protFrame = protFrame[:]

del protFrame[1]
print(protFrame[1][:5])
print(n_protFrame[1][:5])

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


## How we classify and seperate the data

classes = ('No long COVID','Long covid')

# Applying any needed transformations. The tensor transformation is needed because tensors can operate on the GPU and support auto-differentiation
#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# initiailizing our dataset. splitting into training and testing dataset with a variable training ratio

## Experiment regime: 200,400,600,800,1000 epochs, full batch size (using full batch size because the dataset is small anyways). Learning rate: 0.01,0.001,0.0001,0.00001
epochs = [200,400,600,800,1000]
lrs = [0.01,0.001,0.0001,0.00001]

tst = 10
record = np.zeros((tst,len(epochs),len(lrs)))

for c in range(tst):
    count1 = 0
    for q in epochs:
        count2 = 0

        for j in lrs:
            training_ratio = 0.8
            train_protein,test_protein = set_datasets('data/4-4-22_Test.AllLC_vsHealthy_Data.csv','data/4-4-22_Test.AllNLC_vsHealthy_Data.csv',training_ratio)
            for i in range(len(train_protein)):
                sample = train_protein[i]
                print(sample['label'])

            ##
            train_loader = torch.utils.data.DataLoader(train_protein, batch_size=len(train_protein), shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_protein, batch_size=1, shuffle=False)
            print(train_loader)
            print(test_loader)

            ##
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
            ##
            '''
            PATH = './mlp_proteomics_reset_' + str(count1) + str(count2) + '.pth'
            torch.save(mlp.state_dict(), PATH)
            '''
            ##
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(mlp.parameters(), lr=j, momentum=0.9)

            ##
            print(device)
            ##
            '''
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
            
            
            PATH = './mlp_proteomics_' + str(count1) + str(count2) + '.pth'
            torch.save(mlp.state_dict(), PATH)
            '''

            ##
            PATH = './models/epoch_lr/mlp_proteomics_' + str(count1) + str(count2) + '.pth'
            mlp = MLP().to(device)
            mlp.load_state_dict(torch.load(PATH))

            ##
            correct = 0
            total = 0

            t1_start = perf_counter()
            with torch.no_grad():
                for data in test_loader:
                    inputs = data['profile']
                    labels = data['label']
                    labels = labels.to(device)
                    outputs = mlp(inputs)
                    u, predicted = torch.max(outputs.data, 1)
                    total = total + labels.size(0)
                    correct = correct + (predicted == labels).sum().item()
                    pass
            t1_stop = perf_counter()

            print(f'Accuracy: {100 * correct // total} %')
            acc1 = str(100 * correct // total)

            record[c,count1,count2] = 100 * correct // total
            count2 = count2 + 1
            print("Elapsed time during the whole program in seconds:",
                                                    t1_stop-t1_start)
        count1 = count1 + 1

##
mean_trials_record = np.mean(record,0)
mean_lr_record = np.mean(record,2)
mean_lr_record = np.mean(mean_lr_record,0)
mean_epochs_record = np.mean(record,1)
mean_epochs_record = np.mean(mean_epochs_record,0)
print(record)
print(mean_trials_record)
print(mean_lr_record)
print(mean_epochs_record)


