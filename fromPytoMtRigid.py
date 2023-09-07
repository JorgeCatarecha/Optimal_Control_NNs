import torch
import scipy.io as sio
import numpy as np

ts = [0]
# These models have 50561 parameters: D = 4, Width = 128
# When model with grad weight

ws = [2]
grad_weight = 1
'''
grad_weight = 10
ws = [0, 0.1, 1, 4, 6]

grad_weight = 20
ws = [0, 2, 4, 7, 9]
'''
for i, w in enumerate(ws):
  for t in ts:
    # path = './Modelos/RigidTanhL1' + str(w) + 't' + str(t) + '.pth'
    path = './Modelos/RigidTanhL1' + str(w) + 't' + str(t) + 'Grad_W'+ str(grad_weight) + '.pth'
    # Load the PyTorch model weights
    state_dict = torch.load(path)
    # Convert the weights to a format compatible with MATLAB
    weight_dict = {
    'inp_weight': state_dict['inp.weight'].numpy().astype(np.double),
    'inp_bias': state_dict['inp.bias'].numpy().astype(np.double),
    'hidden_0_weight': state_dict['hidden.0.weight'].numpy().astype(np.double),
    'hidden_0_bias': state_dict['hidden.0.bias'].numpy().astype(np.double),
    'hidden_1_weight': state_dict['hidden.1.weight'].numpy().astype(np.double),
    'hidden_1_bias': state_dict['hidden.1.bias'].numpy().astype(np.double),
    'hidden_2_weight': state_dict['hidden.2.weight'].numpy().astype(np.double),
    'hidden_2_bias': state_dict['hidden.2.bias'].numpy().astype(np.double),
    'output_weight': state_dict['output.weight'].numpy().astype(np.double),
    'output_bias': state_dict['output.bias'].numpy().astype(np.double)
    }

    path_mat = './Modelos/RigidTanhL1' + str(w) + 't' + str(t) + 'Grad_W'+ str(grad_weight) + '.mat'
    print(path_mat)
    sio.savemat(path_mat, weight_dict)






