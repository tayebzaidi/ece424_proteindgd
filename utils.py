import numpy as np
import scipy as sp
import scipy.io
from scipy.optimize import minimize

def compute_gradient(theta, sequences, L, q, N, ep):
    params = vector_to_tensor_array(theta, L, q)
    sum_prob_diffs = 0 # sum of all probability differences
    for n in range(N): # for all sequences
        sn = sequences[n,:] # extract nth row
        sum_prob_diffs += compute_adjacent_energy_gradient(sn,q,params)
    return ep*sum_prob_diffs/N

def compute_objective(theta, sequences, L, q, N, ep):
    params = vector_to_tensor_array(theta, L, q)
    sum_prob_diffs = 0
    for n in range(N): # for all sequences
        sn = sequences[n,:] # extract nth row
        sum_prob_diffs += compute_adjacent_energy(sn,q,params)
    return ep*sum_prob_diffs/N

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

def energy_gradient(s,q,params):
    '''
    Function to compute the sum of the differences of energy gradients of sequence s with all adjacent sequences

    Inputs:
        s = np.array, represents sequence of integers 0:(q-1) (NOT CURRENTLY 1:q)
        q = int, length of sequence
        params = list, contains fields and couplings [fields.shape = (L,q), couplings.shape = (L,L,q,q)]
        NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)

    '''
    u = np.copy(s)

    L = s.shape[0] # length of sequence
    fields = np.zeros((L,q))
    couplings = np.zeros((L,L,q,q))
    original_seq_energy = compute_energy_vect(s,q,params)

    for pos in range(L): # for each position to differ in
        for qi in range(q): # for each possible flip in this position
            if s[pos] != qi:
                u[pos] = qi
            else:
                continue

            exp_diff = np.exp(0.5*(original_seq_energy - compute_energy_vect(u,q,params)))

            for j in range(L):
                fields[j,s[j]] += exp_diff
                fields[j,u[j]] -= exp_diff

                for i in range(j):
                    couplings[i,j,s[i],s[j]] += exp_diff
                    couplings[i,j,u[i],u[j]] -= exp_diff


        #Undo the adjacency adjustment at position "pos"
        u[pos] = s[pos]

    gradient_vector = np.concatenate((fields.flatten(), couplings.flatten()))
    return(gradient_vector)

def compute_adjacent_energy_gradient(s, q, params):
    '''
    Function to compute the gradients of the energy function with respect to the fields and couplings parameters.
    
    Inputs:
        s = np.array, represents sequence of integers 1:q (NOT CURRENTLY 0:(q-1))
        q = int, length of sequence
        params = list, contains fields and couplings [fields.shape = (L,q), couplings.shape = (L,L,q,q)]
                NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)
    
    Returns:
        grad_fields = np.array, gradients of the energy function with respect to the fields parameters
        grad_couplings = np.array, gradients of the energy function with respect to the couplings parameters
    '''
    L = s.shape[0]  # length of sequence
    inp_fields = params[0]  # field params
    inp_couplings = params[1]  # coupling params
    
    grad_fields = np.zeros_like(inp_fields)
    grad_couplings = np.zeros_like(inp_couplings)
    
    for j in range(L):  # for each position to differ in
        for qi in range(q):  # for each possible flip in this position
            if qi == s[j]:
                continue  # skip the case where qi = s[j]
            
            energy_diff_j = inp_fields[j, s[j]] - inp_fields[j, qi]  # add the field energy difference
            
            for i in range(j):  # for all positions s.t. i < j
                # add difference from couplings
                energy_diff_j += inp_couplings[i, j, s[i], s[j]] - inp_couplings[i, j, s[i], qi]
            
            for i in range(j+1, L):  # for all positions s.t. j < i
                # add difference from couplings
                # NOTE: indices have to be flipped to only use i<j portion of input couplings
                energy_diff_j += inp_couplings[j, i, s[j], s[i]] - inp_couplings[j, i, qi, s[i]]
            
            prob_diff_j = np.exp(0.5 * energy_diff_j)  # probability difference for position j and value qi
            
            # update gradients for fields
            grad_fields[j, s[j]] -= prob_diff_j
            grad_fields[j, qi] += prob_diff_j
            
            # update gradients for couplings
            for i in range(j):  # for all positions s.t. i < j
                grad_couplings[i, j, s[i], s[j]] -= prob_diff_j
                grad_couplings[i, j, s[i], qi] += prob_diff_j
            
            for i in range(j+1, L):  # for all positions s.t. j < i
                grad_couplings[j, i, s[j], s[i]] -= prob_diff_j
                grad_couplings[j, i, qi, s[i]] += prob_diff_j

    gradient_vector = np.concatenate((grad_fields.flatten(), grad_couplings.flatten()))
    return(-1*gradient_vector)

def compute_energy(s,q,params):
    '''
    Function to compute the energy of a sequence s
    Not efficient for computing KL divergence objective or gradients

    Inputs:
        s = np.array, represents sequence of integers 0:(q-1) (NOT CURRENTLY 1:q)
        q = int, length of sequence
        params = list, contains fields and couplings [fields.shape = (L,q), couplings.shape = (L,L,q,q)]
        NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)

    '''
    L = s.shape[0] # length of sequence
    inp_fields = params[0] # field params
    inp_couplings = params[1] # coupling params

    energy = 0;
    for j in range(L):
        #print(energy, inp_fields, L, j, s[j])
        energy += inp_fields[j,s[j]] # add field energy from position i
        for i in range(j):
            energy += inp_couplings[i,j,s[i],s[j]] # add coupling energy from positions i,j

    return energy

def compute_energy_vect(s, q, params):
    L = s.shape[0]
    inp_fields = params[0]
    inp_couplings = params[1]

    # Compute the field energy
    field_energy = np.sum(inp_fields[np.arange(L), s])

    # Compute the coupling energy
    coupling_energy = np.sum(inp_couplings[np.triu_indices(L, k=1)][:, s[:-1], s[1:]])

    return field_energy + coupling_energy

def tensor_array_to_vector(tensor_array):
    """
    Function to convert params from [inp_fields, inp_couplings] form to single 1D array form

    Inputs:
        tensor_array_form = list, expects [inp_fields, inp_couplings] format where both elements of list are numpy arrays
        NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)
    """
    inp_fields = tensor_array[0]
    inp_couplings = tensor_array[1]
    return np.concatenate((inp_fields.flatten(), inp_couplings.flatten()))

def vector_to_tensor_array(vector, L, q):
    """
    Function to convert params from 1D vector form to [inp_fields, inp_couplings] form

    Inputs:
        vector = np.ndarray, expects [....] format where components can be reshapen into fields and couplings arrays
        NOTE: only part of couplings(i,j,:,:) s.t. i<j is used (i.e. upper tri portion)
    """
    inp_fields = np.reshape(vector[:L*q], (L,q))
    inp_couplings = np.reshape(vector[L*q:], (L,L,q,q))
    return [inp_fields, inp_couplings]

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window