##
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter

from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
import pandas as pd
from torchmetrics import ConfusionMatrix
import seaborn as sn



## Sets the device to the GPU (mps is for apple m1 chip). If using HPC, then change line to:
#  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.has_mps else "cpu")
#print(device)


def z_stand(mat):
    mat_norm = np.mean(mat,0)
    mat_std = np.std(mat,0)
    z_s = (mat - mat_norm)/mat_std
    return z_s

class ProtDataset(Dataset):
    def __init__(self,prot_frame,transform=None):
        self.protFrame = prot_frame
        self.protFrame = np.asarray(self.protFrame)
        self.protprof = self.protFrame[:,6:].astype('float')
        self.protprof = np.delete(self.protprof,8345,1) #only for early testing
        self.protlabel = self.protFrame[:,4].astype('float')

        # z-score standardize the proteomics
        self.protprof = z_stand(self.protprof)
        #print(self.protprof)
        #print(self.protlabel)
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
def set_datasets(csv1): #two csv files for proteomics. Ratio between 0 to 10.
    # processing protein spreadsheets.
    train_protFrame = pd.read_csv(csv1)
    train_protFrame = train_protFrame.values.tolist()

    # Send to pytorch dataset class
    train_set = ProtDataset(train_protFrame)
    return train_set

## initiailizing our dataset. splitting into training and testing dataset with a variable training ratio
n_trials = 50
accuracies = np.zeros(n_trials)
for q in range(n_trials):
    training_ratio = [50,22]
    train_protein = set_datasets('./Prot_Meta_Combined2.csv')
    data_train, data_val = random_split(train_protein, training_ratio)
    #print(data_train[:]['label'])
    #print(data_val[:]['label'])

    ##
    test_loader = torch.utils.data.DataLoader(data_val, batch_size=22, shuffle=False)#, pin_memory=True)

    ##
    class MLP(nn.Module):
      def __init__(self):
        super(MLP, self).__init__()
        # TODO: define your MLP
        self.m0 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(8952, 2048)
        self.m1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(2048, 1024)
        self.m2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1024, 256)
        self.m3 = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(256, 1)

      def forward(self, x):
        # TODO: define your forward function
        #batch_size,features = x.size()
        x = x.type(torch.FloatTensor)
        #x = x.to(device)
        #x = x.view(batch_size,-1)
        x = torch.flatten(x,1)
        x = self.m0(x)
        x = F.relu(self.fc1(x))
        x = self.m1(x)
        x = F.relu(self.fc2(x))
        x = self.m2(x)
        x = F.relu(self.fc3(x))
        x = self.m3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

    #mlp = MLP().to(device)
    mlp = MLP()
    PATH = '/Users/hongb3/PycharmProjects/pythonProject1/lightning_logs_25trials/version_1380301/checkpoints/epoch=2999-step=12000.ckpt'
    checkpoint = torch.load(PATH,map_location=torch.device('cpu')) #, map_location=torch.device('mps')
    mlp.load_state_dict(checkpoint['state_dict'])
    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data['profile']
            labels = data['label']
            count = 0
            #for f in range (10)
            #print(inputs[])
            #print(labels)
            labels = labels.type(torch.IntTensor)
            #labels = labels.to(device)
            outputs = mlp(inputs)
            predicted = torch.round(outputs.data)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()


    acc = 100 * correct // total
    accuracies[q] = acc
    confmat = ConfusionMatrix(num_classes=2)
    testmat = confmat(predicted,labels)
    print(testmat)
    sn.heatmap(testmat,annot=True)
    plt.show()

meanacc = np.mean(accuracies)
print(accuracies)
print('Mean accuracy across 50 trials: ', meanacc)
