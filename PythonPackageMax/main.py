import os, sys

from mpi4py import MPI
import numpy as np

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
data = load_data_for_node(rank,size)

# Initialize your model parameters
model_params = initialize_model_parameters()

# Consensus matrix
P = np.full((size, size), 1.0/size)

# Gradient descent parameters
learning_rate = 0.01
num_iterations = 1000

for iteration in range(num_iterations):
    # Compute local gradients based on local data and model parameters
    local_gradient = compute_gradient(data, model_params)
    
    # Allreduce operation to perform the consensus operation
    # This step replaces sending individual messages and manually computing the consensus
    # consensus_gradient = np.zeros_like(local_gradient)
    # comm.Allreduce(local_gradient, consensus_gradient, op=MPI.SUM)
    
    # # Scale the consensus gradient according to the consensus matrix
    # # For a complete graph and uniform consensus matrix, this is a simple averaging
    # consensus_gradient 
    
    # Update model parameters
    model_params = model_params - learning_rate * local_gradient

    

    # Optional: Check for convergence or perform logging

# Finalize MPI (not strictly necessary in scripts, but good practice)
MPI.Finalize()