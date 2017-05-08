#!/usr/bin/env python
# OSVR Training Review Script
# Port of original MATLAB script
from scipy.io import savemat, loadmat
import numpy as np
from parameterisation import construct_admm_parameters
from constructParams import constructParams

MATLAB_FILENAME = "data.mat"

def load_MAT(var_name):
    loaded = loadmat(file_name=MATLAB_FILENAME, variable_names=[var_name])
    if var_name in loaded:
        return loaded[var_name]
    else:
        print("MATLAB File Load Error")
        return None

# Constants
loss = 2 # Loss function of OSVR
bias = 1 #include bias term or not in OSVR
lam = 1 # (lambda) scaling parameter for primal variables in OSVR
gamma = [100, 1] # Loss balance parameter
smooth = 1 # temporal smoothness on ordinal constraints
epsilon = [0.1, 1] # Parameter in epsilon-SVR
rho = 0.1 # augmented Lagrangian multiplier
flag = 0 # unsupervise learning flag
max_iter = 300 # maximum number of iteration in optimising OSVR

train_data_seq = load_MAT("train_data_seq")
train_label_seq = load_MAT("train_label_seq")
test_data = load_MAT("test_data")
test_label = load_MAT("test_label")

construct_admm_parameters(train_data_seq,train_label_seq,epsilon,bias,flag)
#constructParams(train_data_seq,train_label_seq,epsilon,bias,flag)


