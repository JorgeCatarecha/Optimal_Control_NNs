import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time

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
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

class Net(nn.Module):
  def __init__(self, width, depth):
    """
    width: number of neurons in each hidden layer
    depth: number of layers in the network
    """
    super().__init__()
    self.flatten = nn.Flatten()  # Flattens input
    self.inp = nn.Linear(6, width)  # From input to hidden layer
    self.hidden = nn.ModuleList()  # Define our hidden layers in a list
    for _ in range(depth-1):
      self.hidden.append(nn.Linear(width, width))  # From hidden layer to hidden layer

    self.sig = nn.Tanh()  # ReLU activation layer 
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

class NetReLU(nn.Module):
  def __init__(self, width, depth):
    """
    width: number of neurons in each hidden layer
    depth: number of layers in the network
    """
    super().__init__()
    self.flatten = nn.Flatten()  # Flattens input
    self.inp = nn.Linear(6, width)  # From input to hidden layer
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
 
    x=df.iloc[:,0:6].values
    y=df.iloc[:,6].values
    z=df.iloc[:,7:13].values
 
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
    def __init__(self, gradient_weight):
        super(CustomLoss, self).__init__()
        self.gradient_weight = gradient_weight

    def forward(self, outputs, targets, inputs, targets2):
        mse_loss = nn.MSELoss(reduction='sum')(outputs, targets)
        gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]
        gradient_loss = nn.MSELoss(reduction='sum')(gradients, targets2)
        loss = mse_loss + self.gradient_weight * gradient_loss
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

data = pd.read_csv('./Datasets/Rigid.csv', header=None)
data.columns = ['t', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'V', 'Vx1', 'Vx2','Vx3', 'Vx4','Vx5', 'Vx6', 'Error']
data = data[data['Error']==0].drop('Error', axis=1)
grouped_df = data.groupby('t')
grouped_dict = dict(list(grouped_df))
train_per = 0.8
container = ModelsContainer(grouped_dict, train_per)
test_sets = container.data_test
batch_size = 128

ws = [0, 0.1, 1]
ts = data['t'].unique()
models = [[],[],[],[]]
sparsity = [[],[],[],[]]
l2s = [[],[],[],[]]
h1s = [[],[],[],[]]
crit = CustomLoss(gradient_weight=0)
crit2 = CustomLoss(gradient_weight=1)
grad_weight = 10
for i, w in enumerate(ws):
  for j, t in enumerate(ts):
    test_set = test_sets[j]
    path = './Modelos/Rigid/RigidTanhL1' + str(w) + 't' + str(container.ts[j]) + 'Grad_W'+ str(grad_weight) + '.pth'
    model = Net(width=128, depth=4)
    model.load_state_dict(torch.load(path))
    model.eval()
    models[0].append(model)
    sparsity[0].append(calculate_zero_percentage(model))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    l2s[0].append(test_epoch(testloader, model, crit))
    h1s[0].append(test_epoch(testloader, model, crit2))
    path = './Modelos/Rigid/RigidTanhL1' + str(w) + 't' + str(container.ts[j]) + '.pth'
    model = Net(width=128, depth=4)
    model.load_state_dict(torch.load(path))
    model.eval()
    models[1].append(model)
    sparsity[1].append(calculate_zero_percentage(model))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    l2s[1].append(test_epoch(testloader, model, crit))
    h1s[1].append(test_epoch(testloader, model, crit2))

w = 4
path = './Modelos/Rigid/RigidTanhL1' + str(w) + 't' + str(container.ts[j]) + 'Grad_W'+ str(grad_weight) + '.pth'
model = Net(width=128, depth=4)
model.load_state_dict(torch.load(path))
model.eval()
models[0].append(model)
sparsity[0].append(calculate_zero_percentage(model))
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
l2s[0].append(test_epoch(testloader, model, crit))
h1s[0].append(test_epoch(testloader, model, crit2))
w = 6
path = './Modelos/Rigid/RigidTanhL1' + str(w) + 't' + str(container.ts[j]) + 'Grad_W'+ str(grad_weight) + '.pth'
model = Net(width=128, depth=4)
model.load_state_dict(torch.load(path))
model.eval()
models[0].append(model)
sparsity[0].append(calculate_zero_percentage(model))
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
l2s[0].append(test_epoch(testloader, model, crit))
h1s[0].append(test_epoch(testloader, model, crit2))

ws2 = [0,2,4,7,9]
for i, w in enumerate(ws2):
  for j, t in enumerate(ts):
    test_set = test_sets[j]
    path = './Modelos/Rigid/RigidTanhL1' + str(w) + 't' + str(container.ts[j]) + 'Grad_W'+ str(20) + '.pth'
    model = Net(width=128, depth=4)
    model.load_state_dict(torch.load(path))
    model.eval()
    models[2].append(model)
    sparsity[2].append(calculate_zero_percentage(model))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    l2s[2].append(test_epoch(testloader, model, crit))
    h1s[2].append(test_epoch(testloader, model, crit2))

ws3 = [0, 0.1, 0.5,1]
for i, w in enumerate(ws3):
  for j, t in enumerate(ts):
    test_set = test_sets[j]
    path = './Modelos/Rigid/RigidL1' + str(w) + 't' + str(container.ts[j]) + '.pth'
    model = NetReLU(width=128, depth=4)
    model.load_state_dict(torch.load(path))
    model.eval()
    models[3].append(model)
    sparsity[3].append(calculate_zero_percentage(model))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    l2s[3].append(test_epoch(testloader, model, crit))
    h1s[3].append(test_epoch(testloader, model, crit2))

# Compare errors in test set
plt.figure(1)
plt.plot(ws,l2s[1],'o-', label='Grad_Weight: '+str(1))
plt.plot(ws+[4,6],l2s[0],'^--', label='Grad_Weight: '+str(10))
plt.plot(ws2,l2s[2],'<--', label='Grad_Weight: '+str(20))
plt.xlabel(r'$Penalty L_1$')
plt.ylabel(r'$L^2$ Error')
plt.legend()

plt.figure(2)
plt.plot(ws,h1s[1],'o-',  label='Grad_Weight: '+str(1))
plt.plot(ws+[4,6],h1s[0],'^--', label='Grad_Weight: '+str(10))
plt.plot(ws2,h1s[2],'<--', label='Grad_Weight: '+str(20))
plt.xlabel(r'$Penalty L_1$')
plt.ylabel(r'$H^1$ Error')
plt.legend()

plt.figure(3)
plt.plot(ws,sparsity[1],'o-',  label='Grad_Weight: '+str(1))
plt.plot(ws+[4,6],sparsity[0],'^--', label='Grad_Weight: '+str(10))
plt.plot(ws2,sparsity[2],'<--', label='Grad_Weight: '+str(20))
plt.xlabel(r'$Penalty L_1$')
plt.ylabel(r'Sparsity')
plt.legend()

plt.figure(4)
plt.plot(sparsity[1],l2s[1],'o-', label=r'$\mu_{Grad}$: '+str(1))
plt.plot(sparsity[0],l2s[0],'^--', label='$\mu_{Grad}$: '+str(10))
plt.plot(sparsity[2],l2s[2],'<--', label='$\mu_{Grad}$: '+str(20))
plt.plot(sparsity[3],l2s[3],'<--', label='ReLU $\mu_{Grad}$: '+str(1))
plt.xlabel('Sparsity')
plt.ylabel(r'$L^2$ Error')
plt.legend()

plt.figure(5)
plt.plot(sparsity[1],h1s[1],'o-',  label='$\mu_{Grad}$: '+str(0))
plt.plot(sparsity[0],h1s[0],'^--', label='$\mu_{Grad}$: '+str(10))
plt.plot(sparsity[2],h1s[2],'<--', label='$\mu_{Grad}$: '+str(20))
plt.plot(sparsity[3],h1s[3],'<--', label='ReLU $\mu_{Grad}$: '+str(1))
plt.xlabel('Sparsity')
plt.yscale('log')
plt.ylabel(r'$H^1$ Error')
plt.legend()
plt.show()
print(sparsity)