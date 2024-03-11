def compute_gradient():
    params = vector_to_tensor_array(theta, L, q)
    sum_prob_diffs = 0; # sum of all probability differences
    for n in range(N): # for all sequences
        sn = sequences[n,:] # extract nth row
        sum_prob_diffs += energy_gradient(sn,q,params)
    return ep*sum_prob_diffs/N

def compute_objective():
    params = vector_to_tensor_array(theta, L, q)
    sum_prob_diffs = 0;
    for n in range(N): # for all sequences
        sn = sequences[n,:] # extract nth row
        sum_prob_diffs += compute_adjacent_energy(sn,q,params)
    return ep*sum_prob_diffs/N

def load_data_for_node(rank,size):
    return 1

def initialize_model_parameters():
    return 1

def compute_adjacent_energy(s,q,params):
    '''
    Function to compute energy differences of all adjacent sequences to s and then sum the results. i.e.

    sum_{u~s} exp[(1/2)*(Energy(s; params) - Energy(u; params))]

    Inputs:
        s = np.array, represents sequence of integers 1:q (NOT CURRENTLY 0:(q-1))
        q = int, length of sequence
        params = list, contains fields and couplings [fields.shape = (L,q), couplings.shape = (L,L,q,q)]
        NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)

    '''
    L = s.shape[0] # length of sequence
    inp_fields = params[0] # field params
    inp_couplings = params[1] # coupling params

    sum_prob_diff_total = 0; # total energy difference summed over adjacent sequences

    for j in range(L): # for each position to differ in
        for qi in range(q): # for each possible flip in this position
            energy_diff_j = inp_fields[j,s[j]] - inp_fields[j,qi] # add the field energy difference
            # NOTE: one of the above terms added will be zero when q = s[j]-1. Not a problem.

            for i in range(j): # for all positions s.t. i < j
                # add difference from couplings
                energy_diff_j += inp_couplings[i,j,s[i],s[j]] - inp_couplings[i,j,s[i],qi]

            for i in range(j+1,L): # for all positions s.t. j < i
                # add difference from couplings
                # NOTE: indices have to be flipped to only use i<j portion of input couplings
                energy_diff_j += inp_couplings[j,i,s[j],s[i]] - inp_couplings[j,i,qi,s[i]]

            sum_prob_diff_total += np.exp(0.5*energy_diff_j) # add probability difference to total sum

    sum_prob_diff_total -= 1 # corrects for case where qi = s[j]-1, which is added above

    return(sum_prob_diff_total)

def gradient(theta, sequences, L, q, N, ep):
    params = vector_to_tensor_array(theta, L, q)
    sum_prob_diffs = 0; # sum of all probability differences
    for n in range(N): # for all sequences
        sn = sequences[n,:] # extract nth row
        sum_prob_diffs += energy_gradient(sn,q,params)
    return ep*sum_prob_diffs/N
