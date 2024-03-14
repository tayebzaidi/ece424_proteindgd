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

def run_dgd_complete(comm, file_path, total_seqs, n_local_grad_steps):
    # MPI initialization
    size = comm.Get_size() # Total number of processes
    rank = comm.Get_rank() # The current process ID

    start_time = MPI.Wtime()   

    # Assuming your data is somehow distributed or replicated across nodes
    # Load your data here. This could vary greatly depending on your application.
    sequence_data = load_data_for_node(comm, rank, size, file_path, total_seqs)
    #print(sequence_data[0:2,:])

    # Initialize your model parameters
    L = sequence_data.shape[1] # sequence length
    N = sequence_data.shape[0] # number of sequences
    q = np.max(sequence_data)+1 #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
    ep = 1 # this will just be a multiplier on the gradient descent step

    #if rank == 0:
    #    print("Completing DGD on sequence of L={} q={}".format(L,q))


    alpha = 1e-2; # stepsize
    max_iter = int(1e3); # maximum iterations (consensus steps)
    tol = 1e-6; # tolerance criterion on objective differences

    # start with initial parameter set
    init_fields = np.zeros((L,q))
    init_couplings = np.zeros((L,L,q,q))
    init_params = [init_fields, init_couplings]
    init_theta = tensor_array_to_vector(init_params)

    theta_k = np.copy(init_theta)

    # Number of independent steps before consensus
    n_steps = n_local_grad_steps

    # Initialize objective values
    obj_val_prev = 1e9 #Large value
    obj_values = np.zeros(max_iter*n_steps)
    avg_diff_values = []

    # Distributed gradient descent loop
    for iteration in range(max_iter):
        # Independent gradient descent steps on each node
        for step in range(n_steps):
            # Compute local gradients based on local data and model parameters
            local_gradient = compute_gradient(theta_k, sequence_data, L, q, N, ep)

            # Perform line search to determine the step size
            res = sp.optimize.line_search(compute_objective, compute_gradient, theta_k, -local_gradient,
                                        args=(sequence_data, L, q, N, ep))
            alpha = res[0]
            # Update the previous objective value
            obj_values[iteration*n_steps+step] = res[4]
            #print("Node {}: Alpha: {:.4f}".format(rank, alpha))

            # Update model parameters using the local step size
            theta_k -= alpha * local_gradient

            if rank == 0:
                # Print the iteration and average objective value difference
                if step % 2 == 0:
                    local_obj_diff = np.abs(obj_values[iteration*n_steps+step] - obj_values[iteration*n_steps+step-1])
                    print("Local Gradient Step: {}, Objective Value: {:.7f}".format(step, local_obj_diff))
            

        # Perform all-reduce to compute the sum of updated parameters across all processes
        consensus_theta = np.zeros_like(theta_k)
        comm.Allreduce(theta_k, consensus_theta, op=MPI.SUM)

        # Scale the consensus parameters according to the consensus matrix
        # For a complete graph and uniform consensus matrix, this is a simple averaging
        theta_k = consensus_theta / size

        # Compute the current and previous objective values (from the line search just run previously) and gather differences
        obj_val_curr = compute_objective(theta_k, sequence_data, L, q, N, ep)
        obj_val_diff = np.abs(obj_val_prev - obj_val_curr)
        obj_val_diffs = comm.gather(obj_val_diff, root=0)

        if rank == 0:
            # Compute the average objective value difference across all nodes
            avg_obj_val_diff = np.mean(obj_val_diffs)
            print("Consensus Step: {}, Iteration: {}, Average Objective Value Difference: {:.7f}".format(iteration, iteration*(n_steps), avg_obj_val_diff))

            # Check for convergence
            if avg_obj_val_diff < tol:
                print("Convergence reached. Average objective value difference: {:.7f}".format(avg_obj_val_diff))


        else:
            avg_obj_val_diff = None

        # Broadcast the break condition to all nodes
        avg_obj_val_diff = comm.bcast(avg_obj_val_diff, root=0)
        avg_diff_values.append(avg_obj_val_diff)
        break_condition = avg_obj_val_diff < tol

        if break_condition:
            print("Breaking at node {}".format(rank))
            break

        obj_val_prev = obj_val_curr

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

        print(norm_couplings)

    #     plt.imshow(norm_couplings)
    #     plt.colorbar()

    #     ax = plt.gca()
    #     xticks = [0,1,2,3]
    #     yticks = xticks
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(yticks)
    #     plt.show()

    #     plt.plot(avg_diff_values[1:])
    #     plt.show()

    # plt.plot(np.trim_zeros(obj_values,'b'))
    # plt.plot(moving_average(np.trim_zeros(obj_values,'b'),n_steps))
    # plt.title("Node: {}".format(rank))
    # plt.show()
        
    end_time = MPI.Wtime()
    total_time = end_time - start_time

    consensus_iterations = iteration + 1

    return total_time, consensus_iterations

def run_dgd_ring(comm, file_path, total_seqs, n_local_grad_steps):
    # MPI initialization
    size = comm.Get_size() # Total number of processes
    rank = comm.Get_rank() # The current process ID
    # Determine neighbors in the ring topology
    left_neighbor = (rank - 1) % size
    right_neighbor = (rank + 1) % size

    start_time = MPI.Wtime()   

    # Assuming your data is somehow distributed or replicated across nodes
    # Load your data here. This could vary greatly depending on your application.
    sequence_data = load_data_for_node(comm, rank, size, file_path, total_seqs)
    #print(sequence_data[0:2,:])

    # Initialize your model parameters
    L = sequence_data.shape[1] # sequence length
    N = sequence_data.shape[0] # number of sequences
    q = np.max(sequence_data)+1 #hardcoded, did not save MUST SAVE IN NEW BENCHMARKS
    ep = 1 # this will just be a multiplier on the gradient descent step

    #if rank == 0:
    #    print("Completing DGD on sequence of L={} q={}".format(L,q))


    alpha = 1e-2; # stepsize
    max_iter = int(1e3); # maximum iterations (consensus steps)
    tol = 1e-6; # tolerance criterion on objective differences

    # start with initial parameter set
    init_fields = np.zeros((L,q))
    init_couplings = np.zeros((L,L,q,q))
    init_params = [init_fields, init_couplings]
    init_theta = tensor_array_to_vector(init_params)

    theta_k = np.copy(init_theta)

    # Number of independent steps before consensus
    n_steps = n_local_grad_steps

    # Initialize objective values
    obj_val_prev = 1e9 #Large value
    obj_values = np.zeros(max_iter*n_steps)
    avg_diff_values = []

    # Distributed gradient descent loop
    for iteration in range(max_iter):
        # Independent gradient descent steps on each node
        for step in range(n_steps):
            # Compute local gradients based on local data and model parameters
            local_gradient = compute_gradient(theta_k, sequence_data, L, q, N, ep)

            # Perform line search to determine the step size
            res = sp.optimize.line_search(compute_objective, compute_gradient, theta_k, -local_gradient,
                                        args=(sequence_data, L, q, N, ep))
            alpha = res[0]
            # Update the previous objective value
            obj_values[iteration*n_steps+step] = res[4]
            #print("Node {}: Alpha: {:.4f}".format(rank, alpha))

            # Update model parameters using the local step size
            theta_k -= alpha * local_gradient

            if rank == 0:
                # Print the iteration and average objective value difference
                if step % 2 == 0:
                    local_obj_diff = np.abs(obj_values[iteration*n_steps+step] - obj_values[iteration*n_steps+step-1])
                    print("Local Gradient Step: {}, Objective Value: {:.7f}".format(step, local_obj_diff))
            

        # Prepare buffers for non-blocking sends and receives
        recv_buffer_left = np.zeros_like(theta_k)
        recv_buffer_right = np.zeros_like(theta_k)
        reqs = []

        # Non-blocking send to the right neighbor and non-blocking receive from the left neighbor
        reqs.append(comm.Isend(theta_k, dest=right_neighbor))
        reqs.append(comm.Irecv(recv_buffer_left, source=left_neighbor))
        
        # Non-blocking send to the left neighbor and non-blocking receive from the right neighbor
        reqs.append(comm.Isend(theta_k, dest=left_neighbor))
        reqs.append(comm.Irecv(recv_buffer_right, source=right_neighbor))

        # Wait for all non-blocking operations to complete
        MPI.Request.Waitall(reqs)

        # Average parameters from left, right, and self
        theta_k = (theta_k + recv_buffer_left + recv_buffer_right) / 3

        # Compute the current and previous objective values (from the line search just run previously) and gather differences
        obj_val_curr = compute_objective(theta_k, sequence_data, L, q, N, ep)
        obj_val_diff = np.abs(obj_val_prev - obj_val_curr)
        obj_val_diffs = comm.gather(obj_val_diff, root=0)

        if rank == 0:
            # Compute the average objective value difference across all nodes
            avg_obj_val_diff = np.mean(obj_val_diffs)
            print("Consensus Step: {}, Iteration: {}, Average Objective Value Difference: {:.7f}".format(iteration, iteration*(n_steps), avg_obj_val_diff))

            # Check for convergence
            if avg_obj_val_diff < tol:
                print("Convergence reached. Average objective value difference: {:.7f}".format(avg_obj_val_diff))


        else:
            avg_obj_val_diff = None

        # Broadcast the break condition to all nodes
        avg_obj_val_diff = comm.bcast(avg_obj_val_diff, root=0)
        avg_diff_values.append(avg_obj_val_diff)
        break_condition = avg_obj_val_diff < tol

        if break_condition:
            print("Breaking at node {}".format(rank))
            break

        obj_val_prev = obj_val_curr

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

        print(norm_couplings)

    #     plt.imshow(norm_couplings)
    #     plt.colorbar()

    #     ax = plt.gca()
    #     xticks = [0,1,2,3]
    #     yticks = xticks
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(yticks)
    #     plt.show()

    #     plt.plot(avg_diff_values[1:])
    #     plt.show()

    # plt.plot(np.trim_zeros(obj_values,'b'))
    # plt.plot(moving_average(np.trim_zeros(obj_values,'b'),n_steps))
    # plt.title("Node: {}".format(rank))
    # plt.show()
        
    end_time = MPI.Wtime()
    total_time = end_time - start_time

    consensus_iterations = iteration + 1

    return total_time, consensus_iterations


if __name__ == "__main__":
    file_path = '../Experimentation/full_align_L10_q6.mat'
    total_seqs = 8192
    n_steps = 15
    graph_structure = 'complete'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    if rank == 0:
        print("Solving: Structure -- {}, Total Seqs -- {}, File -- {}, Local Steps -- {},".format(
            graph_structure, total_seqs, file_path[-10:-4], n_steps))

    if graph_structure == 'ring':
        [total_time, consensus_iterations] = run_dgd_ring(comm, file_path, total_seqs, n_steps)
    elif graph_structure == 'complete':
        [total_time, consensus_iterations] = run_dgd_complete(comm, file_path, total_seqs, n_steps)

    rank = comm.Get_rank()

    if rank == 0:
        print(f"Total time taken by rank 0: {total_time:.2f} seconds")
        print("Consensus iterations required: {}, Total iterations: {}".format(consensus_iterations, consensus_iterations*(n_steps+1)+1))

    MPI.Finalize()