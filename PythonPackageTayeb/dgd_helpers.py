import scipy as sp

def load_data_for_node(comm, rank, size):
    print(rank)

    data_to_process = None

    if rank == 0:
        data = sp.io.loadmat('../Experimentation/full_align_L6_q3(1).mat')
        sequence_data = data["full_align"] # numpy array of sequences
        sequence_data -= 1; # to account for python vs. matlab starting integers
    
        ##Apportion correct amounts of data to each node
        total_seqs = 1024
        seq_per_node = total_seqs // size # Assume evenly divisible (power of twos only)
        data_chunks = [sequence_data[i* seq_per_node:(i+1) * seq_per_node] for i in range(size)]
        print(data_chunks)
    else:
        data_chunks = None
    
    data_subsection = comm.scatter(data_chunks, root=0)
    return data_subsection

def initialize_model_parameters():
    return 1