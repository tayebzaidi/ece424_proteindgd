
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy as sp
from utils import *
# from dgd_helpers import *

# parent_dir = os.path.join(os.path.dirname(__file__), '..')
# normalized_path = os.path.normpath(parent_dir)
# sys.path.append(normalized_path)

# import data
data = sp.io.loadmat('./Experimentation/full_align_L6_q3.mat')
sequence_data = data["full_align"] # numpy array of sequences
sequence_data -= 1; # to account for python vs. matlab starting integers
sequence_data = sequence_data[:100,:]

L = sequence_data.shape[1] # sequence length
N = sequence_data.shape[0] # number of sequences
q = 3; #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
ep = 1; # this will just be a multiplier on the gradient descent step


alpha = 1e-2; # stepsize
max_iter = int(1e3); # maximum iterations
tol = 1e-9; # tolerance criterion on gradient norm
output_level = 2;   

# start with initial parameter set
init_fields = np.zeros((L,q))
init_couplings = np.zeros((L,L,q,q))
init_params = [init_fields, init_couplings]
init_theta = tensor_array_to_vector(init_params)

# compute initial grad, initial objective
# note: there is something fishy going on if I use the version in utils.py
obj_val = compute_objective(init_theta, sequence_data, L, q, N, ep)
init_grad = compute_gradient(init_theta, sequence_data, L, q, N, ep)
# norm_grad_fields = np.linalg.norm(init_grad[0])
# norm_grad_couplings = np.linalg.norm(init_grad[1])
norm_theta_k = np.linalg.norm(init_theta,np.inf)

# format output
if output_level >= 2:
    output_header = '%6s %6s %12s %6s' % \
        ('iter', 'f', '||Grad_theta||','alpha')
    print(output_header)
    print('%6i %9.2e %9.2e %9.2e' %
            (0, obj_val, norm_theta_k, alpha))

# iteration variables
theta_k = init_theta

# for k in range(max_iter):
for k in range(max_iter):
    
    # compute gradient at iterate
    grad_k = compute_gradient(theta_k, sequence_data, L, q, N, ep)
    
    # compute step size
    res = sp.optimize.line_search(compute_objective, compute_gradient, theta_k, -grad_k, 
                                  args=(sequence_data, L, q, N, ep))
    alpha = res[0]
    
    # take step
    theta_k -= alpha*grad_k

    # compute norms
    obj_val = compute_objective(theta_k, sequence_data, L, q, N, ep)
    norm_theta_k = np.linalg.norm(init_theta,np.inf)
    
    # check convergence criteria
    if norm_theta_k < tol:
        print("Iteration converged successfully to tolerance")
        print('%6i %9.2e %9.2e %9.2e' %
        (0, obj_val, norm_theta_k, alpha))
    
        break
        
    # print iterations
    if output_level >= 2 and k % 10 == 0:
    # Print the output header every 100 iterations
        if k % 100 == 0:
            print(output_header)
        print('%6i %9.2e %9.2e %9.2e' %
                (0, obj_val, norm_theta_k, alpha))
    

    




