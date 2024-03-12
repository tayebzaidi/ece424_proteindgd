
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy as sp
from utils import *
import matplotlib.pyplot as plt
# from dgd_helpers import *

# import data
data = sp.io.loadmat('./Experimentation/full_align_L6_q3_maxtest.mat')
sequence_data = data["full_align"] # numpy array of sequences
sequence_data -= 1; # to account for python vs. matlab starting integers
sequence_data = sequence_data[:1000,:]


L = sequence_data.shape[1] # sequence length
N = sequence_data.shape[0] # number of sequences
q = 3; #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
ep = 1; # this will just be a multiplier on the gradient descent step


alpha = 1e-2; # stepsize
max_iter = int(5e2); # maximum iterations
tol = 1e-8; # tolerance criterion on gradient norm
output_level = 2;  

# start with initial parameter set
init_fields = np.zeros((L,q))
init_couplings = np.zeros((L,L,q,q))
init_params = [init_fields, init_couplings]
init_theta = tensor_array_to_vector(init_params)

# compute initial grad, initial objective
obj_val = compute_objective(init_theta, sequence_data, L, q, N, ep)
init_grad = compute_gradient(init_theta, sequence_data, L, q, N, ep)
norm_theta_k = np.linalg.norm(init_theta,np.inf)

# format output header
if output_level >= 2:
    output_header = '%6s %9s %12s %6s' % \
        ('iter', 'f', '||D_obj||','alpha')
    print(output_header)

# iteration variables
theta_k = init_theta # problem - this vector is too long 

# f_args = [sequence_data, L, q, N, ep] # used in automatic gradient computation

# for k in range(max_iter):
for k in range(max_iter):
    
    # compute gradient at iterate
    grad_k = compute_gradient(theta_k, sequence_data, L, q, N, ep)
    # grad_k = sp.optimize.approx_fprime(theta_k,compute_objective, 1e-7, *f_args)
    
    # compute step size via line search
    res = sp.optimize.line_search(compute_objective, compute_gradient, theta_k, -grad_k, 
                                args=(sequence_data, L, q, N, ep))
    alpha = res[0]

    # take step
    if alpha is None:
        print("Line search did not return valid step size")
        break
    else:
        theta_k -= alpha*grad_k

    # compute norms
    obj_val_new = compute_objective(theta_k, sequence_data, L, q, N, ep)
    delta_f = abs(obj_val_new - obj_val)
    obj_val = obj_val_new
    # norm_theta_k = np.linalg.norm(theta_k,np.inf)
    
    # check convergence criteria NOTE: conv criteria on norm of gradient doesnt work? 
    if delta_f < tol:
        print('Iteration converged successfully to tolerance')
        break
        
    # print iterations
    if output_level >= 2 and k % 1 == 0:
    # Print the output header every 10 iterations
        if (k+1) % 10 == 0:
            print(output_header)
        print('%6i %9.2e %12.2e %9.2e' %
                (k, obj_val, delta_f, alpha))
        
if output_level >= 1:
    print('Final objective.................: %g' % obj_val)
    print('|Delta_obj| at final point.........: %g' % delta_f)
    print('Number of iterations............: %d' % k)
    print('')

# display results
results = vector_to_tensor_array(theta_k, L, q)
out_fields = results[0]
out_couplings = results[1]

norm_couplings = np.linalg.norm(out_couplings, ord='fro', axis=(2,3))
norm_couplings = norm_couplings + norm_couplings.T

plt.imshow(norm_couplings)
plt.colorbar()

ax = plt.gca()
xticks = [0,1,2,3]
yticks = xticks
ax.set_xticks(xticks)
ax.set_yticks(yticks)

plt.show()
    




