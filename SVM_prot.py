##
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from torch.utils.data import Dataset
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

##
def svm(traindata, trainlabels, testdata, C=1, kernel_type='linear', gamma='scale'):
  # TODO: implement svm
  vc = SVC(C = C, kernel = kernel_type, gamma=gamma) #, max_iter = 100
  vc.fit(traindata,trainlabels)
  preds = vc.predict(testdata)
  return preds

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




n_trials = 20
accs = np.zeros((n_trials,4))
for i in range(n_trials):
    ## initiailizing our dataset. splitting into training and testing dataset with a variable training ratio
    training_ratio = 0.8
    train_protein, test_protein = set_datasets('data/4-4-22_Test.AllLC_vsHealthy_Data.csv',
                                               'data/4-4-22_Test.AllNLC_vsHealthy_Data.csv', training_ratio)

    ##
    train = train_protein[:]['profile']
    labels = train_protein[:]['label']
    pvalid = test_protein[:]['profile']
    validlabels = test_protein[:]['label']
    Cparam = [0.5, 1.0, 5.0, 10.0]
    predac = np.ones(len(Cparam))
    count = 0;

    for cp in Cparam:

      pred = svm(train,labels,pvalid, C = cp, kernel_type='rbf', gamma='scale')

      # TODO: evaluate classification accuracy
      accuracy = validlabels - pred
      accuracy = accuracy[accuracy == 0]
      accuracy = len(accuracy) / len(validlabels)
      accs[i,count] = accuracy
      count = count + 1

    print(accs)

    # plt.figure()
    # plt.plot(Cparam, predac)
    # plt.title('Accuracy of Valid Data Predictions vs. Changing C parameter')
    # plt.xlabel('C parameter')
    # plt.ylabel('Proportion or correct predictions')
    # plt.show()
