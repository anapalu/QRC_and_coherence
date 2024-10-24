from itertools import combinations

import numpy as np
from numpy import linalg as la
from qrccoherence.njitfuncts import VN_entropy, fast_kron
from qrccoherence.pauli_matrices import Sy


def partial_trace(rh, N, indices2keep):
    finaldim = int(2**len(indices2keep))
    dims0 = tuple([2]*N*2)
    reshaped_dm = rh.reshape(dims0)
    qbs_list = list(range(N))
    [qbs_list.remove(x) for x in indices2keep]
    qbs_list.reverse()
    n = N
    for qb2trace in qbs_list:
        reshaped_dm = np.trace(reshaped_dm, axis1=qb2trace, axis2=qb2trace + n)
        n = n - 1
    return reshaped_dm.reshape(finaldim, finaldim)

def partial_transpose(rho, N, indices2keep):
    """
    Function essentially copied from qutip. mask is a list with length qutipdims[0] containing booleans 
    or 0s and 1s indicating which directions to transpose. indices2keep are the ones that are NOT to be 
    transposed.
    """
    mask = [1] * N
    for x in indices2keep:
        mask[x] = 0
    
    initial_shape_rho = np.shape(rho)
    qt_dims = [[2]*N]*2
    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                            [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])
    data = rho.reshape(
        np.array(qt_dims).flatten()).transpose(pt_idx).reshape(initial_shape_rho)
    return data

def reduced_product_rho_np(rho, N):
    product_rho = partial_trace(rho, N, indices2keep=[0])
    for i in range(1, N):
        rh = partial_trace(rho, N, indices2keep=[i])
        product_rho = fast_kron(product_rho, rh).reshape(2**(1+i), 2**(1+i))
    return product_rho

def decohering_operation(rho):
    return np.diag(np.diag(rho))

def vN_entropy(rho):
    eigvals, _ = la.eigh(rho)
    s=0
    for val in eigvals:
        eigenvalue = np.round(val, 6)
        if eigenvalue != 0:
            s -= val*np.log(val)
    return np.round(s, 6)

def information_content(rho, N):
    rh_tensor = reduced_product_rho_np(rho, N)
    decohered_rh_tensor = decohering_operation(rh_tensor)
    decohered_rho = decohering_operation(rho)

    S_rho = VN_entropy(rho)#vN_entropy(rho)
    S_rh_tensor = VN_entropy(rh_tensor)#vN_entropy(rh_tensor)
    S_decohered_rh_tensor = VN_entropy(decohered_rh_tensor)#vN_entropy(decohered_rh_tensor)
    S_decohered_rho = VN_entropy(decohered_rho)#vN_entropy(decohered_rho)

    total_mutual_info = S_rh_tensor - S_rho
    local_coherence = S_decohered_rh_tensor - S_rh_tensor
    total_coherence = S_decohered_rho - S_rho
    hookup = total_mutual_info + local_coherence
    return total_mutual_info, local_coherence, total_coherence, hookup


sysy = fast_kron(Sy, Sy).reshape(4, 4)
def concurrence(rho, N):
    c = []
    for (i, j) in combinations(range(N), 2):
        rho_reduced = partial_trace(rho, N, indices2keep = [i, j])
        rhotilde = sysy @ np.conj(rho_reduced) @ sysy
        evals, P = la.eigh(rho_reduced)
        sqrtrho = P @ np.diag(np.sqrt(evals)) @ np.conj(P).T
        R2 = sqrtrho @ rhotilde @ sqrtrho
        evals, _ = la.eigh(R2)
        sqrtevals = np.sqrt(evals)
        maxeval = np.max(sqrtevals)
        c.append(np.max([0, 2 * maxeval - np.sum(sqrtevals)]))
    return np.mean(c), np.std(c)

def negativity(rho, N):
    neg = []
    neg1 = []
    for (i, j) in combinations(range(N), 2):
        rho_PT = partial_transpose(rho, N, [i, j])
        ## calculate trace norm
        rho2 = rho_PT @ np.conj(rho_PT).T
        evals, _ = la.eigh(rho2)
        tracenorm_PT = np.sum(np.sqrt(evals))
        
        neg.append(0.5 * (tracenorm_PT - 1))
    for i in range(N):
        rho_PT = partial_transpose(rho, N, [i])
        ## calculate trace norm
        rho2 = rho_PT @ np.conj(rho_PT).T
        evals, _ = la.eigh(rho2)
        tracenorm_PT = np.sum(np.sqrt(evals))
        
        neg1.append(0.5 * (tracenorm_PT - 1))
    negtot = list(np.copy(neg))
    negtot.extend(neg1)
    
    return np.mean(neg), np.std(neg), np.mean(neg1), np.std(neg1), np.mean(negtot), np.std(negtot)
