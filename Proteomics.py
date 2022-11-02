##
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from torch.utils.data import Dataset
import pandas as pd


## Sets the device to the GPU (mps is for apple m1 chip)
device = torch.device("mps" if torch.has_mps else "cpu")
print(device)

def z_stand(mat):
    mat_norm = np.mean(mat,0)
    mat_std = np.std(mat,0)
    z_s = (mat - mat_norm)/mat_std
    return z_s

## Dataset and loader.
class ProtDataset(Dataset):
    def __init__(self,prot_frame,transform=None):
        self.protFrame = prot_frame
        self.protFrame = np.asarray(self.protFrame)
        self.protprof = self.protFrame[:,7:].astype('float')
        self.protlabel = self.protFrame[:,5].astype('float')

        # z-score standardize the proteomics
        self.protprof = z_stand(self.protprof)

        self.transform = transform

    def __len__(self):
        return len(self.protFrame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #pat_id = self.protFrame[idx][2]
        pat_label = self.protlabel[idx]
        pat_profile = self.protprof[idx,:]
        sample = {'label':pat_label,'profile': pat_profile}

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

## initiailizing our dataset. splitting into training and testing dataset with a variable training ratio
n_trials = 10
accuracies = np.zeros(n_trials)
for q in range(n_trials):
    training_ratio = 0.8
    train_protein,test_protein = set_datasets('data/4-4-22_Test.AllLC_vsHealthy_Data.csv','data/4-4-22_Test.AllNLC_vsHealthy_Data.csv',training_ratio)

    ##
    train_loader = torch.utils.data.DataLoader(train_protein, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_protein, batch_size=1, shuffle=False)

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
    PATH = './models/basic/mlp_reset.pth'
    torch.save(mlp.state_dict(), PATH)

    ##
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters()) #for sgd: ,lr=0.001,momentum=0.9

    ##
    n_epoch = 40
    mlp.train()
    t1_start = perf_counter()
    for epoch in range(n_epoch):  # loop over the dataset multiple times
      for i, data in enumerate(train_loader, 0):
        # TODO: write training code
        inputs = data['profile']
        #print(inputs)
        labels = data['label']
        #print(labels)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        outputs = np.squeeze(outputs)
        #print(outputs)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:",
                                            t1_stop-t1_start)

    PATH = './models/basic/mlp.pth'
    torch.save(mlp.state_dict(), PATH)

    ##
    PATH = './models/basic/mlp.pth'
    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load(PATH))
    mlp.eval()

    correct = 0
    total = 0
    label_rec = np.zeros(14)
    predicted_rec = np.zeros(14)
    counter = 0
    t1_start = perf_counter()
    with torch.no_grad():
        for data in test_loader:
            inputs = data['profile']
            labels = data['label']
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            label_rec[counter] = labels
            outputs = mlp(inputs)
            predicted = torch.round(outputs.data)
            predicted_rec[counter]=predicted
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
            counter = counter + 1
            pass
    t1_stop = perf_counter()
    print(label_rec)
    print(predicted_rec)
    acc = 100 * correct // total
    print(f'Accuracy: {acc} %')

    accuracies[q] = acc
    print(accuracies)