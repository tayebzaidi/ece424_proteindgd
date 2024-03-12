
import numpy as np
import scipy as sp
from utils import * 
from dgd_helpers import *
import os

# import data
data = sp.io.loadmat('./Experimentation/full_align_L6_q3.mat')
sequence_data = data["full_align"] # numpy array of sequences
sequence_data -= 1; # to account for python vs. matlab starting integers
sequence_data = sequence_data[:100,:]

L = sequence_data.shape[1] # sequence length
N = sequence_data.shape[0] # number of sequences
q = 3; #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
ep = 1; # this will just be a multiplier on the gradient descent step

alpha = 1e-3; # stepsize
max_iter = 1e3; # maximum iterations
tol = 1e-3; # tolerance criterion on gradient norm

# start with initial parameter set
init_fields = 0.1*np.ones((L,q))
init_couplings = 0.1*np.ones((L,L,q,q))
init_params = [init_fields, init_couplings]

# compute initial grad, initial objective
init_f = objective(init_params, sequence_data, L, q, N, ep)
init_grad = gradient(init_params, sequence_data, L, q, N, ep)

# print(sequence_data[1,:])
# test = gradient_compute_adjacent_energy(sequence_data[1,:], q, init_params)
# print(test)

# # print output header
# if output_level >= 2:
#     # Print the output header every 10 iterations
#     if iter_count % 10 == 0:
#         print(output_header)
#     print('%6i %9.2e %9.2e %9.2e %5i %9i' %
#             (iter_count, f_k, norm_pk, alpha_k, changed_Wk, W_k[W_k].shape[0]))


# for k in range(max_iter):

# for steps in DGD:

    # compute gradient

    # compute step length (?)




