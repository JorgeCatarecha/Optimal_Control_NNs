import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
# from tomlkit import ws

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('./Datasets/Rigid_Riccati.csv', header=None)
data.columns = ['t', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'V', 'Vx1', 'Vx2','Vx3', 'Vx4','Vx5', 'Vx6',
                'Vx1x1', 'Vx1x2','Vx1x3', 'Vx1x4','Vx1x5', 'Vx1x6',
                'Vx2x1', 'Vx2x2','Vx2x3', 'Vx2x4','Vx2x5', 'Vx2x6',
                'Vx3x1', 'Vx3x2','Vx3x3', 'Vx3x4','Vx3x5', 'Vx3x6',
                'Vx4x1', 'Vx4x2','Vx4x3', 'Vx4x4','Vx4x5', 'Vx4x6',
                'Vx5x1', 'Vx5x2','Vx5x3', 'Vx5x4','Vx5x5', 'Vx5x6',
                'Vx6x1', 'Vx6x2','Vx6x3', 'Vx6x4','Vx6x5', 'Vx6x6',
                'Error']

ts = data['t'].unique()
data = data[data['Error']==0].drop('Error', axis=1)
grouped_df = data.groupby('t')
grouped_dict = dict(list(grouped_df))

class MyDataset(torch.utils.data.Dataset):
  # Given a dataframe provides a dataset instance, splitting, input, output and gradient
  def __init__(self,df):
 
    x=df.iloc[:,0:6].values
    y=df.iloc[:,6].values
    z=df.iloc[:,7:13].values
    hes=df.iloc[:,13:49].values
 
    self.x_train=torch.tensor(x,dtype=torch.float32).to(device)
    self.y_train=torch.tensor(y,dtype=torch.float32).to(device)
    self.z_train=torch.tensor(z,dtype=torch.float32).to(device)
    self.hes_train=torch.tensor(hes,dtype=torch.float32).to(device)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx], self.y_train[idx], self.z_train[idx], self.hes_train[idx]
  
class ModelsContainer():
  def __init__(self, dictionary, train_num):
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
      ind = train_num
      self.Ns.append(N)
      self.data_train.append(MyDataset(df.iloc[0:ind]))
      self.data_test.append(MyDataset(df.iloc[ind:N]))

def prox(x, alpha):
    return torch.sign(x) * torch.relu(torch.abs(x) - alpha)


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

class Net(nn.Module):
  def __init__(self, width, depth):
    '''
    width: number of neurons in each hidden layer
    depth: number of hidden layers in the network
    '''
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
  
def vect_hess(fun, input):
    hessian = []
    for i in input:
        hessian.append(torch.autograd.functional.hessian(fun, i))
    hessian_tensor =  torch.stack(hessian) 
    return hessian_tensor

class CustomLoss(nn.Module):
    # To fit the gradient as well
    def __init__(self, gradient_weight, hess_weight):
        super(CustomLoss, self).__init__()
        self.gradient_weight = gradient_weight
        self.hess_weight = hess_weight

    def forward(self, outputs, targets, inputs, targets2, model, targets3):
        mse_loss = nn.MSELoss(reduction='sum')(outputs, targets)
        gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]
        gradient_loss = nn.MSELoss(reduction='sum')(gradients, targets2)
        hess = vect_hess(model, inputs).flatten(1,2)
        hess_loss = nn.MSELoss(reduction='sum')(hess, targets3)
        loss = mse_loss + self.gradient_weight * gradient_loss + self.hess_weight * hess_loss
        return loss

class CustomLoss2(nn.Module):
    # To fit the gradient as well
    def __init__(self, gradient_weight):
        super(CustomLoss2, self).__init__()
        self.gradient_weight = gradient_weight

    def forward(self, outputs, targets, inputs, targets2, model, targets3):
        mse_loss = nn.MSELoss(reduction='sum')(outputs, targets)
        gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]
        gradient_loss = nn.MSELoss(reduction='sum')(gradients, targets2)
        loss = mse_loss + self.gradient_weight * gradient_loss
        return loss
    
def train_epoch(trainloader, net, optimizer, criterion, l1_weight):
  '''
  trainloader: training set split in batches
  net: net to train
  optimizer: we can use SGD or ADAM
  criterion: loss function used, we use Cross Entropy

  output: average criterion over the points in the training set
  '''
  loss_c = 0  # Cumulative loss
  n_total = 0  # Number of points in set

  for x_batch, y_batch, z_batch, hes_batch in trainloader:
    x_batch.requires_grad=True
    n_total += len(x_batch)
    # Predict output
    preds = net.forward(x_batch)
    # Compute loss
    loss = criterion(preds, y_batch.unsqueeze(1), x_batch, z_batch, net, hes_batch)
    # Back-propagate and update parameters
    optimizer.zero_grad()
    loss.backward()
    loss_c += loss.item()  # Update cumulative loss
    optimizer.step()
    if l1_weight != 0:
      step_size = optimizer.param_groups[0]['lr']
      with torch.no_grad():
        for param in net.parameters():
          param.data = prox(param.data, alpha=l1_weight*step_size)

  train_loss = loss_c/n_total  # Compute average
  return train_loss

def test_epoch(testloader, net, criterion):
  '''
  Input:
  testloader: test set split in batches
  net: model considered
  criterion: loss function used, we use Cross Entropy

  Output:
  - average loss over the points in the test set
  - error (proportion) over the points in the test set 
  '''
  loss_c = 0  # Cumulative loss
  n_total = 0  # Number of points
  
  for x_batch, y_batch, z_batch, hes_batch in testloader:
    x_batch.requires_grad=True
    pred = net.forward(x_batch)  # predict V
    loss = criterion(pred, y_batch.unsqueeze(1), x_batch, z_batch, net, hes_batch)
    loss_c += loss.item()  # cumulative loss
    n_total += len(x_batch)  # Add number of points in this batch

  avrg_test_loss = loss_c/n_total
  return avrg_test_loss

def set_run(net, train_set, test_set, batch_size, lr, n_epochs, criterion, criterion2, optimizer_class, l1_weight=0, crit3=0):
  '''
  Input:
  net: model
  batch_size: size of batches considered when splitting our dataset
  lr: learning rate used in the optimizer
  n_epochs: number of epochs used to train our network
  criterion: loss function used: Cross Entropy
  optimizer_class: optimizer used, either SGD or Adam

  Output:
  loss_train_list: list of train loss for each epoch
  loss_test_list: list of test loss for each epoch
  err_test_list: list of test error for each epoch
  '''

  trainloader, testloader = loading_data(batch_size, train_set, test_set)  # Load dataset
  optimizer = optimizer_class(net.parameters(), lr = lr)  # Initialise optimizer

  # Initialise output values
  loss_train_list = []
  loss_test_list = []

  for epoch in range(n_epochs):
    net.train()
    train_loss = train_epoch(trainloader, net, optimizer, criterion, l1_weight)  # Train for epoch
    # Compute and print lossess and errors
    net.eval()
    test_loss = test_epoch(testloader, net, criterion2)
    print(f'Epoch: {epoch+1:03} | Train Loss: {train_loss:.04} | Test Loss  (H1 but fitting hessian): {test_loss:.04} ')
    # Record values in the output lists
    loss_train_list.append(train_loss)
    loss_test_list.append(test_loss)
  if crit3 != 0:
    extra_error = test_epoch(testloader, net, crit3)
    return loss_train_list, loss_test_list, extra_error
  return loss_train_list, loss_test_list

def calculate_zero_percentage(model):
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param==0).sum().item()

    zero_percentage = (zero_params / total_params)

    return zero_percentage

SEED = 2325122  # CID is 02325122
np.random.seed(SEED)
torch.manual_seed(SEED) 

grad_weight = 10
hess_weight = 10/6
crit3 = CustomLoss2(gradient_weight=0)
crit2 = CustomLoss2(gradient_weight=1)
crit = CustomLoss(gradient_weight=grad_weight, hess_weight=hess_weight)
opt = optim.SGD # Use SGD
ws = [0]
train_nums = [3000, 5000]
for train_num in train_nums:
    container = ModelsContainer(grouped_dict, train_num)
    for w in ws:
      test_losses = []
      l2s = []
      sparsity = []
      for i in range(len(container.ts)):
        net = Net(width=128, depth=4).to(device) # Width*(4 + (Depth-+1)*Width)
        data_train = container.data_train[i]
        data_test = container.data_test[i]
        _, _ = set_run(net, data_train, data_test, batch_size=100, lr=0.00001, n_epochs=50, criterion=crit, criterion2=crit2, optimizer_class=opt, l1_weight=w)
        _, _ = set_run(net, data_train, data_test, batch_size=100, lr=0.000001, n_epochs=1800, criterion=crit, criterion2=crit2, optimizer_class=opt, l1_weight=w)
        _, loss_test_list2b, extra = set_run(net, data_train, data_test, batch_size=100, lr=0.0000001, n_epochs=1, criterion=crit, criterion2=crit2, optimizer_class=opt, l1_weight=w, crit3 = crit3)
        path = './Modelos/PointCompRigid/RigidHessL1' + str(w) + 't' + str(container.ts[i]) + 'Grad_W'+ str(grad_weight) + 'Grad_H'+ str(hess_weight) + 'Points'+ str(train_num) +'.pth'
        torch.save(net.state_dict(), path)
        container.models.append(net)
        test_losses.append(loss_test_list2b[0])
        sparsity.append(calculate_zero_percentage(net))
        l2s.append(extra)
    
plt.figure(1)
plt.plot(w, sparsity,'^--')
plt.plot(w, test_losses,'o--')
plt.plot(w, l2s,'o--')
plt.legend(['Sparsity Proportion', 'Test H1 Loss', 'Test L2 Loss'])
plt.xlabel('Initial Time')


## Plot points and surface
inds = [0]

for i in inds:
    t = container.ts[i]
    df = grouped_dict[t]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    net = container.models[i]

    # Make data.
    X = np.arange(-np.pi/3, np.pi/3+0.05, 0.05)
    Y = np.arange(-np.pi/4, np.pi/4+0.05, 0.05)

    X, Y = np.meshgrid(X, Y)
    other = np.zeros(shape=X.shape)
    xy = np.concatenate([X[:,:,np.newaxis],other[:,:,np.newaxis],other[:,:,np.newaxis],Y[:,:,np.newaxis],other[:,:,np.newaxis],other[:,:,np.newaxis]],axis=2)
    xy = xy.reshape(-1, xy.shape[-1])
    Z = net(torch.tensor(xy).float())

    Z = Z.detach().numpy().reshape(X.shape)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False, alpha=.8)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('phi')
    ax.set_ylabel('omega 1')
    ax.set_zlabel('V')
    ax.set_title(f't0 = {t:.2f}')

"""
auxiliar = torch.tensor([[0.,0.]], requires_grad=True)
val = container.models[8](auxiliar)
print(val)
grad = torch.autograd.grad(outputs=val, inputs=auxiliar, grad_outputs=torch.ones_like(val), create_graph=True)
print(grad)
"""
plt.show() 