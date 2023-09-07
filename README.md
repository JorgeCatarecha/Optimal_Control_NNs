We provide the following codes:

MATLAB:
- PMP_Solver_Lobatto, PMP_Solver_Marching, PMP_Solver_Shooting, are different
solvers for the PMP system for a certain control problem. We recommend using 
PMP_Solver_Marching which is an extension of PMP_Solver_Lobatto including a 
time-marching step.
- Data_generation and Data_generation_VdP_Riccati generate datasets for the Van
der Pol problem without and with the hessian. Analogously, Data_generation_Solid
and Data_generation_Solid_Riccati, do the corresponding with the rigid body 
attitude problem.
- load_model_from_py and load_model_from_py_tanh_rigid are functions that take
the path of the .mat file of a NN and returns it as a MATLAB function whose 
output is the value $V$ and its gradient. The first one for models with ReLU
as activation function, and the second one for models with tanh. You can also
provide a boolean parameter to indicate if you want the sparse implementation of
the matrices or the dense one.
- time_comparison and time_comparison_rigid use the previous files, to load model
and measure the evaluation time of the models. They correspond to models of
VdP problem and the rigid body one.
- control_trajectory_comparison and control_trajectory_comparison_rigid take
different models and an initial condition to recover the optimal control and 
trajectories. They also compare them with the optimal and uncontrolled ones. 
The first file do it for the VdP problem, and the second one for the Rigid one.

Python:
- NN-Grad-VdP, NN-Grad-Rigid, NN-Hess-VdP and NN-Hess-Rigid, use a dataset to 
train Neural Networks using first (Grad) or second (Hess) order information to
approximate the function $V$ of the corresponding problem: VdP (Van der Pol) or
Rigid (Rigid Body Problem). They can also plot the approximation $V$ and save 
the models as .pth files.
- fromPytoMt and fromPytoMtRigid are functions that take the path of these
.pth files of the models and transform them to .mat files so that MATLAB can
read them. They do it for VdP and Rigid.
-retrain and retrain-Hess, are functions that take the path of the .pth of models
and train them again using a certain dataset and first or second order 
information.
- Load_Model_VdP_Comp_Points and Load_Model_Rigid are files that take the path
of several models and a dataset and compare the models performance over it. They
can compare models with different penalties, number of points used during 
training, etc.