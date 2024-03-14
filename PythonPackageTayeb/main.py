import os, sys

from mpi4py import MPI
import numpy as np
import json
import subprocess

import matplotlib.pyplot as plt

#Local Imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
normalized_path = os.path.normpath(parent_dir)
sys.path.append(normalized_path)
from utils import *
from dgd_helpers import *


if __name__ == "__main__":
    num_processor_list = [1,2,4,8]

    for num_processors in num_processor_list:
        command = f"python benchmarking_functions.py {num_processors}"
        mpi_command = f"mpirun -np {num_processors} {command}"
        print(mpi_command)
        process = subprocess.Popen(mpi_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()


#     file_path = '../Experimentation/full_align_L10_q10.mat'
#     total_seqs = 8192
#     n_steps = 15
#     graph_structure = 'complete'
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()


#     if rank == 0:
#         print("Solving: Structure -- {}, Total Seqs -- {}, File -- {}, Local Steps -- {},".format(
#             graph_structure, total_seqs, file_path[-10:-4], n_steps))

#     if graph_structure == 'ring':
#         [total_time, consensus_iterations] = run_dgd_ring(comm, file_path, total_seqs, n_steps)
#     elif graph_structure == 'complete':
#         [total_time, consensus_iterations] = run_dgd_complete(comm, file_path, total_seqs, n_steps)

#     rank = comm.Get_rank()

#     if rank == 0:
#         print(f"Total time taken by rank 0: {total_time:.2f} seconds")
#         print("Consensus iterations required: {}, Total iterations: {}".format(consensus_iterations, consensus_iterations*(n_steps+1)+1))

#     MPI.Finalize()