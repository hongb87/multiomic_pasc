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

## Sets the device to the GPU (mps is for apple m1 chip)
device = torch.device("mps" if torch.has_mps else "cpu")
print(device)

## Dataset and loader

# How we classify and seperate the data
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# Applying any needed transformations. The tensor transformation is needed because tensors can operate on the GPU and support auto-differentiation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# initiailizing our dataset. splitting into training and testing dataset
cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform)
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform)
print("Print the training dataset:\n ", cifar10_trainset)
print("Print the testing dataset:\n ", cifar10_testset)

print(np.shape((cifar10_trainset[1][0]))) # the trainset has 2 groups: one is the images and one is the associated labels

##
train_loader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_testset, batch_size=1, shuffle=False)
print(train_loader)
print(test_loader)

##
def cifar_imshow(img):
  img = img / 2 + 0.5     # normalize
  npimg = img.numpy()
  return np.transpose(npimg, (1, 2, 0))

print(np.shape(cifar10_trainset[0][0]))

fig, axs = plt.subplots(5, 5, figsize = (12, 12))
plt.gray()

# loop through subplots and add mnist images
for i, ax in enumerate(axs.flat):
  ax.imshow(cifar_imshow(cifar10_trainset[i][0]), cmap=cm.gray_r)
  ax.axis('off')
  ax.set_title('Class {}'.format(classes[cifar10_trainset[i][1]]))

##
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # TODO: define your MLP
    self.m1 = nn.Dropout(p=0.3)
    self.fc1 = nn.Linear(3072, 2048)
    self.m2 = nn.Dropout(p=0.3)
    self.fc2 = nn.Linear(2048, 1024)
    self.m3 = nn.Dropout(p=0.3)
    self.fc3 = nn.Linear(1024, 512)
    self.fc4 = nn.Linear(512, 10)

  def forward(self, x):
    # TODO: define your forward function
    x = x.to(device)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.m2(x)
    x = F.relu(self.fc2(x))
    x = self.m3(x)
    x = F.relu(self.fc3(x))
    x = F.softmax(self.fc4(x))
    return x

mlp = MLP().to(device)
print(mlp)
##
PATH = './mlp_cifar10_reset.pth'
torch.save(mlp.state_dict(), PATH)

##
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)

##
print(device)
##
n_epoch = 5

for epoch in range(n_epoch):  # loop over the dataset multiple times
  for i, data in enumerate(train_loader, 0):
    # TODO: write training code

    inputs, labels = data
   # inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = mlp(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

PATH = './mlp_cifar10.pth'
torch.save(mlp.state_dict(), PATH)

##
PATH = './mlp_cifar10.pth'
mlp = MLP().to(device)
mlp.load_state_dict(torch.load(PATH))

##
correct = 0
total = 0

t1_start = perf_counter()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        labels = labels.to(device)
        outputs = mlp(inputs)
        u, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        pass
t1_stop = perf_counter()

print(f'Accuracy: {100 * correct // total} %')
acc1 = str(100 * correct // total)
print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)