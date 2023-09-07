import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)
class Net(nn.Module):
  def __init__(self, width, depth):
    '''
    width: number of neurons in each hidden layer
    depth: number of layers in the network
    '''
    super().__init__()
    self.flatten = nn.Flatten()  # Flattens input
    self.inp = nn.Linear(2, width)  # From input to hidden layer
    self.hidden = nn.ModuleList()  # Define our hidden layers in a list
    for _ in range(depth-1):
      self.hidden.append(nn.Linear(width, width))  # From hidden layer to hidden layer

    self.sig = nn.Tanh()  # Tanh activation layer 
    self.output = nn.Linear(width, 1)  # From hidden layer to output

  def forward(self, input):
    # Input: matrix of samples to compute output of the network
    # We apply previously defined layers
    x = self.inp(input)
    x = self.sig(x)
    for layer in self.hidden:  # Use a loop to apply layers in self.hidden list
      x = layer(x)
      x = self.sig(x)
    output = self.output(x)
    return output

class NetRelu(nn.Module):
  def __init__(self, width, depth):
    '''
    width: number of neurons in each hidden layer
    depth: number of layers in the network
    '''
    super().__init__()
    self.flatten = nn.Flatten()  # Flattens input
    self.inp = nn.Linear(2, width)  # From input to hidden layer
    self.hidden = nn.ModuleList()  # Define our hidden layers in a list
    for _ in range(depth-1):
      self.hidden.append(nn.Linear(width, width))  # From hidden layer to hidden layer

    self.sig = nn.ReLU()  # ReLU activation layer 
    self.output = nn.Linear(width, 1)  # From hidden layer to output

  def forward(self, input):
    # Input: matrix of samples to compute output of the network
    # We apply previously defined layers
    x = self.inp(input)
    x = self.sig(x)
    for layer in self.hidden:  # Use a loop to apply layers in self.hidden list
      x = layer(x)
      x = self.sig(x)
    output = self.output(x)
    return output

def calculate_zero_percentage(model):
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param==0).sum().item()

    zero_percentage = (zero_params / total_params)
    print(f'Number of parameters: {total_params}. Number of non-zero parameters: {total_params-zero_params}.')
    return zero_percentage

class MyDataset(torch.utils.data.Dataset):
  # Given a dataframe provides a dataset instance, splitting, input, output and gradient
  def __init__(self,df):
 
    x=df.iloc[:,0:2].values
    y=df.iloc[:,2].values
    z=df.iloc[:,3:5].values
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
    self.z_train=torch.tensor(z,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx], self.y_train[idx], self.z_train[idx]

def loading_data(batch_size, train_set, test_set):
  '''
  Input:
  batch_size: size of each of the batches we are splitting our set into
  train_set: set used for training
  test_set: set used for testing
  Output:
  trainloader: train set split into batches with size equal to batch_size
  testloader: test set split into batches with size equal to batch_size
  '''
  trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
  return trainloader, testloader

class ModelsContainer():
  def __init__(self, dictionary, train_per):
    self.ts = []
    self.data = dictionary.values()
    self.models = []
    self.data_train = []
    self.data_test = []
    self.Ns = []
    for item in dictionary.items():
      self.ts.append(item[0])
      df = item[1].drop('t', axis=1)
      N = df.shape[0]
      ind = int(np.round(N*train_per))
      self.Ns.append(N)
      self.data_train.append(MyDataset(df.iloc[0:ind]))
      self.data_test.append(MyDataset(df.iloc[ind:N]))

class CustomLoss(nn.Module):
    # To fit the gradient as well
    def __init__(self, gradient_weight, value_weight=1):
        super(CustomLoss, self).__init__()
        self.gradient_weight = gradient_weight
        self.value_weight = value_weight

    def forward(self, outputs, targets, inputs, targets2):
        mse_loss = nn.MSELoss(reduction='sum')(outputs, targets)
        gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]
        gradient_loss = nn.MSELoss(reduction='sum')(gradients, targets2)
        loss =  self.value_weight * mse_loss + self.gradient_weight * gradient_loss
        return loss

def test_epoch(testloader, net, criterion):
  """
  Input:
  testloader: test set split in batches
  net: model considered
  criterion: loss function used, we use Cross Entropy

  Output:
  - average loss over the points in the test set
  - error (proportion) over the points in the test set 
  """
  loss_c = 0  # Cumulative loss
  n_total = 0  # Number of points
  
  for x_batch, y_batch, z_batch in testloader:
    x_batch.requires_grad=True
    pred = net.forward(x_batch)  # predict V
    loss = criterion(pred, y_batch.unsqueeze(1), x_batch, z_batch)
    loss_c += loss.item()  # cumulative loss
    n_total += len(x_batch)  # Add number of points in this batch

  avrg_test_loss = loss_c/n_total
  return avrg_test_loss

def prox(x, alpha):
    return torch.sign(x) * torch.relu(torch.abs(x) - alpha)

data = pd.read_csv('./Datasets/VanDerPol.csv', header=None)
data.columns = ['t', 'x1', 'x2', 'V', 'Vx1', 'Vx2', 'Error']
data = data[data['Error']==0].drop('Error', axis=1)
grouped_df = data.groupby('t')
grouped_dict = dict(list(grouped_df))
train_per = 0
container = ModelsContainer(grouped_dict, train_per)
test_sets = container.data_test
batch_size = 128

w = 0
ts = [0]
train_nums = [20, 25, 30, 50, 75, 100]
models = [[] for _ in range(2)]
sparsity = [[] for _ in range(2)]
l2s = [[] for _ in range(2)]
h1s = [[] for _ in range(2)]
crit = CustomLoss(gradient_weight=0)
crit2 = CustomLoss(gradient_weight=1)
ws = [0]
w = 0
for train_num in train_nums:
  for i, t in enumerate(ts):
    for j in range(2):
        test_set = test_sets[i]
        if j == 0:
            path = './Modelos/PointComp/VdPHess' + str(w) + 't' + str(container.ts[i]) + 'Points' + str(train_num) + 'Double.pth'
        else:
            path = './Modelos/PointComp/VdP' + str(w) + 't' + str(container.ts[i]) + 'Points' + str(train_num) + 'Double.pth'
        model = Net(width=64, depth=2)
        test_set = test_sets[i]
        model.load_state_dict(torch.load(path))
        model.eval()
        models[j].append(model)
        sparsity[j].append(calculate_zero_percentage(model))
        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        l2s[j].append(test_epoch(testloader, model, crit))
        h1s[j].append(test_epoch(testloader, model, crit2))
lab = ['Grad + Hess', 'Grad']
# Compare errors in test set
plt.figure(1)
for i in range(2):
    plt.plot(train_nums,l2s[i],'o-', label=lab[i])
plt.yscale('log')
plt.xlabel(r'Points in Training')
plt.ylabel(r'$L^2$ Error')
plt.legend()

plt.figure(2)
for i in range(2):
    plt.plot(train_nums,h1s[i],'o-', label=lab[i])
plt.xlabel(r'Points in Training')
plt.yscale('log')
plt.ylabel(r'$H^1$ Error')
plt.legend()

plt.figure(3)
for i in range(2):
    plt.plot(train_nums,sparsity[i],'o-', label=lab[i])
plt.xlabel(r'Points in Training')
plt.ylabel(r'Sparsity')
plt.legend()

plt.show()