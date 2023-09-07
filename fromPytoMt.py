import torch
import scipy.io as sio
import numpy as np

ws = [0, 0.1, 1, 2, 4, 6, 10]
ts = [0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]

for i, w in enumerate(ws):
  for t in ts:
    path = './Modelos/SmallBigL1' + str(w) + 't' + str(t) + '.pth'
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
    'output_weight': state_dict['output.weight'].numpy().astype(np.double),
    'output_bias': state_dict['output.bias'].numpy().astype(np.double)
    }

    path_mat = './Modelos/SmallBigL1' + str(w) + 't' + str(t) + '.mat'
    print(path_mat)
    sio.savemat(path_mat, weight_dict)






