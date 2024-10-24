import numpy as np
from itertools import permutations, combinations

def get_delayed_sk(sk, max_delay):
    delayed_sk = np.empty((max_delay +1, len(sk)))
    delayed_sk[0] = sk
    for i in np.arange(1, max_delay+1):
        delayed_sk[i] = np.hstack((np.zeros(i), sk))[:-i]
    return delayed_sk


def find_sum(nsum): #decomposes a given number in two summands, from the most even decomposition to the most uneven
    res = np.empty(0, dtype=int)
    for i in np.arange(0, nsum):
        for j in np.arange(0, nsum//2+1):
            if i+j == nsum:
                res = np.append(res, (i,j))
    return res.reshape((len(res)//2, 2))


def find_all_sums(nsum, nvar): ###returns all the possible combinations of _nvar_ numbers that add up to _nsum_ 
    if nvar == 1: ##Just one summand, return nsum directly 
        return np.array([np.array([nsum])], dtype = int)
    
    if nvar == nsum: ## If we want the maximum decomposition, return array of ones straightaway 
        return np.array([np.ones(nvar)], dtype = int)
    
    possible = np.empty(0, dtype = int)
    decomposition_in_two = find_sum(nsum) #begin decompositions from 2 summands
    for ns in np.sort(decomposition_in_two): #start decomposing larger numbers first
        summands = np.empty(0, dtype = int)
        summands = np.append(summands, ns)
        it = 2
        while it != nvar:
            summands = np.append(summands[:-1], find_sum(summands[-1])[0]) #keep on decomposing until nvar is reached
            it += 1
            summands = np.sort(summands) #rearrange in order to decompose the largest one in the next interation
            
        if it == nvar: #store once we have the desired number of summands
            possible = np.append(possible, summands)
    possible = possible.reshape((len(decomposition_in_two), nvar))   
    return np.unique(possible, axis = 0)
    
    

def combinations_ordered_by_window(combs):
    windows = np.array([max(arr) - min(arr) + 1 for arr in combs])
    j = 0
    c = np.empty((len(windows), len(combs[0])), dtype=int)
    k = 0
    while j <= max(windows) + 1:
        for i in range(len(windows)):
            if windows[i] == j:
                c[k] = combs[i]
                k += 1
        j += 1
    return c    
    
    
       
def get_all_decompositions(nsum):
    return [find_all_sums(nsum, i+1) for i in range(nsum)]


def get_perm_in_order_basis(perm):
    nv = len(perm)
    order_basis = np.empty(nv, dtype = int)
    for j in range(nv):
        i = 0
        while perm[j] != sorted(perm)[i]:
            i += 1
        order_basis[j] = i
    return order_basis

#### for the monstruous loop:

def separate_pos_by_minidegs(pow_list, positions): #returns a dictionary with each distinct minideg as key and the
                                                    #corresponding positions as values
    nvar = len(pow_list)
    pow_list2 = np.append(pow_list, 0) # add a 0 at the end in orther to close the algorithm
    poss2append = []
    repeat_check = {}
    for j in range(nvar):
        poss2append.append(positions[j])
        if pow_list2[j+1] != pow_list2[j]: ##if the next degree is different, append what we already have
            repeat_check.update({pow_list2[j]:poss2append})
            poss2append = [] #reset poss2append
    return repeat_check



def admissible_orderings(power_list, positions):
    distinct_minidegs = np.unique(power_list) #get dictionary keys
    
    all_dicts = []
    d = {i : 0 for i in distinct_minidegs} ##dummy dict for initialisation, we will remove it at the end
    all_dicts.append(d)
    
    for ordering in permutations(positions, len(power_list)): #all possible orderings of positions
        
        newd = separate_pos_by_minidegs(power_list, ordering) ##dictionary of this particular ordering, 
                                                                ##groups by the minideg they belong to
        non_repeated_count = 0
        for d in all_dicts[::-1]: #run through all the orderings we have already stored, last additions first 
                                    #(for improved efficiency in the search)
                
            if np.array([d[j] == sorted(newd[j]) for j in distinct_minidegs]).all():
                ##we enter here if we are to dismiss this ordering
                break
            non_repeated_count += 1

        if non_repeated_count == len(all_dicts): ## include newd if it's different from previous dicts
            all_dicts.append(newd)
            
    ##discard the dummy
    all_dicts = all_dicts[1:] 
    
    ##rearrange into matrix
    ad_list = np.empty(0) 
    for d in all_dicts:
        s = np.empty(0)
        
        for j in distinct_minidegs:
            s = np.append(s, np.asarray(d[j]))
        ad_list = np.append(ad_list, s)
        
    ad = ad_list.reshape(len(ad_list)//len(power_list), len(power_list))
    
    return ad ##matrix with orderings already stacked
