import os, sys

from mpi4py import MPI
import numpy as np

import matplotlib.pyplot as plt

#Local Imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
normalized_path = os.path.normpath(parent_dir)
sys.path.append(normalized_path)
from utils import *
from dgd_helpers import *

# MPI initialization
comm = MPI.COMM_WORLD
size = comm.Get_size() # Total number of processes
rank = comm.Get_rank() # The current process ID

# Assuming your data is somehow distributed or replicated across nodes
# Load your data here. This could vary greatly depending on your application.
sequence_data = load_data_for_node(comm, rank,size)

# Initialize your model parameters
L = sequence_data.shape[1] # sequence length
N = sequence_data.shape[0] # number of sequences
q = 3; #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
ep = 1; # this will just be a multiplier on the gradient descent step


alpha = 1e-2; # stepsize
max_iter = int(1e3); # maximum iterations
tol = 1e-8; # tolerance criterion on gradient norm

# start with initial parameter set
init_fields = np.zeros((L,q))
init_couplings = np.zeros((L,L,q,q))
init_params = [init_fields, init_couplings]
init_theta = tensor_array_to_vector(init_params)

theta_k = np.copy(init_theta)

## Logging of the objective value for rank 0
if rank == 0:
    objective_values = np.zeros([max_iter])

# Consensus matrix
P = np.full((size, size), 1.0/size)

for iteration in range(max_iter):
    # Compute local gradients based on local data and model parameters
    local_gradient = compute_gradient(theta_k, sequence_data, L, q, N, ep)
        
    # Update model parameters
    theta_k -= alpha * local_gradient

    consensus_theta = np.zeros_like(theta_k)
    comm.Allreduce(theta_k, consensus_theta, op=MPI.SUM)

    # Scale the consensus parameters according to the consensus matrix
    # For a complete graph and uniform consensus matrix, this is a simple averaging
    theta_k = consensus_theta / size

    if rank == 0:
        objective_values[iteration] = compute_objective(theta_k, sequence_data, L, q, N, ep)


# Prepare a container on rank 0 to receive the reduced sum of model parameters
if rank == 0:
    consensus_theta = np.empty_like(theta_k)
else:
    consensus_theta = None 

comm.Reduce(theta_k, consensus_theta, op=MPI.SUM, root=0)

if rank == 0:
    consensus_theta = consensus_theta / size
    results = vector_to_tensor_array(consensus_theta, L, q)
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

    plt.plot(objective_values)
    plt.show()


# Finalize MPI (not strictly necessary in scripts, but good practice)
MPI.Finalize()