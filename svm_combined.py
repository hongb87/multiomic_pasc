##
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,random_split
import pickle

##
def svm(traindata, trainlabels, testdata, C=1, kernel_type='linear', gamma='scale'):
  # TODO: implement svm
  vc = SVC(C = C, kernel = kernel_type, gamma=gamma) #, max_iter = 100
  vc.fit(traindata,trainlabels)
  #with open('testfile.pkl','wb') as pickle_file:
  #  pickle.dump(vc,pickle_file)
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
        self.protprof = self.protFrame[:,6:].astype('float')
        self.protlabel = self.protFrame[:,4].astype('float')

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
def set_datasets(csv1): #two csv files for proteomics. Ratio between 0 to 10.
    # processing protein spreadsheets.
    train_protFrame = pd.read_csv(csv1)
    train_protFrame = train_protFrame.values.tolist()

    # Send to pytorch dataset class
    train_set = ProtDataset(train_protFrame)
    return train_set




n_trials = 25
accs = np.zeros((n_trials,4))
for i in range(n_trials):
    ## initiailizing our dataset. splitting into training and testing dataset with a variable training ratio
    full_dataset = set_datasets('./Prot_Meta_Combined2.csv')
    train_protein, test_protein = random_split(full_dataset, [24,10])

    ##
    train = train_protein[:]['profile']
    labels = train_protein[:]['label']
    pvalid = test_protein[:]['profile']
    validlabels = test_protein[:]['label']
    Cparam = [0.5, 1.0, 5.0, 10.0]
    predac = np.ones(len(Cparam))
    count = 0;

    for cp in Cparam:

      pred = svm(train,labels,pvalid, C = cp, kernel_type='sigmoid', gamma='scale')

      # TODO: evaluate classification accuracy
      accuracy = validlabels - pred
      accuracy = accuracy[accuracy == 0]
      accuracy = len(accuracy) / len(validlabels)
      accs[i,count] = accuracy
      count = count + 1

    print(accs)
    print(np.mean(accs,axis=0))

    # plt.figure()
    # plt.plot(Cparam, predac)
    # plt.title('Accuracy of Valid Data Predictions vs. Changing C parameter')
    # plt.xlabel('C parameter')
    # plt.ylabel('Proportion or correct predictions')
    # plt.show()
