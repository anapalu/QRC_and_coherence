from itertools import combinations

import numpy as np

###DEFINE PAULI MATRICES
Sx = np.array([[0.,1.], [1.,0.]], dtype = 'complex128')
Sy = np.array([[0.,-1j], [1j,0.]], dtype = 'complex128')
Sz = np.array([[1.,0.], [0.,-1.]], dtype = 'complex128')
I = np.array([[1.,0.], [0.,1.]], dtype = 'complex128')

def Sigma(coord, n_spin, N):
    S = [Sz, Sx, Sy]
    s = S[coord]
    sigma = s if n_spin == 0 else I
    for j in range(1, N):
        sigma = np.kron(sigma, s).reshape(2**(j+1), 2**(j+1)) if j == n_spin else np.kron(sigma, I).reshape(2**(j+1), 2**(j+1))
    return sigma

def Ham(J, h, N):   ##GET HAMILTONIAN --> h is an N-component vector, Js is a 2^Nx2^N matrix
    H = 0
    for i in range (N):
        Sx_i = Sigma(1, i, N)
        Sz_i = Sigma(0, i, N)
        for j in range (i):
            Sx_j = Sigma(1, j, N)
            SxiSxj = Sx_i @ Sx_j
            H += J[i,j] * SxiSxj
        H += h[i] * Sz_i
    return H

def Ham_classical_ising(J, h, N):   ##GET HAMILTONIAN --> h is an N-component vector, Js is a 2^Nx2^N matrix
    H = 0 * Sigma(0, 0, N)
    for i in range (N):
        Sz_i = Sigma(0, i, N)
        for j in range (i):
            Sz_j = Sigma(0, j, N)
            SziSzj = Sz_i @ Sz_j
            H += J[i,j] * SziSzj
        H += h[i] * Sz_i
    return H

def generate_nonlocal_obs(N):
    SigmaX = [Sigma(1, x, N) for x in range(N)]
    SigmaY = [Sigma(2, x, N) for x in range(N)]
    SigmaZ = [Sigma(0, x, N) for x in range(N)]
    single_paulis = [SigmaZ, SigmaX, SigmaY]
    O = []
    for i in combinations(np.arange(N*3), 2):
        s0, s1 = i
        spinnum0, spinnum1 = s0 % N, s1 % N
        c0, c1 = s0 // N, s1 // N
        if spinnum0 != spinnum1:
            obs = single_paulis[c0][spinnum0] @ single_paulis[c1][spinnum1]
            O.append(obs)
    return O

def generate_nonlocal_obs_dict(N):
    nonlocalobs_dict = {}
    k = 0
    fromnumcoord2latin = ['z', 'x', 'y']
    for i in combinations(np.arange(N*3), 2):
        s0, s1 = i
        spinnum0, spinnum1 = s0 % N, s1 % N
        c0, c1 = s0 // N, s1 // N
        if spinnum0 != spinnum1:
            nonlocalobs_dict[k] = (fromnumcoord2latin[c0]+ str(spinnum0), fromnumcoord2latin[c1]+str(spinnum1) )
            k += 1
    return nonlocalobs_dict


def get_indices_segregated_measured_obs_matrix(nonlocalobs_dict):
    """
    Returns two lists of indices of the X matrix (nonloc observables), corresponding to strictly classical and to quantum observables.
    """
    columns2keep_zz = []
    columns2keep_xxxyyxyy = []
    columns2keep_xx = []
    for key, val in nonlocalobs_dict.items():
        if val[0][0] == 'z' and val[1][0] == 'z':
            columns2keep_zz.append(key + 1)  # first column of measured observables is all ones for the independent offset
        if val[0][0] != 'z' and val[1][0] != 'z':
            columns2keep_xxxyyxyy.append(key + 1)
        if val[0][0] == 'x' and val[1][0] == 'x':
            columns2keep_xx.append(key + 1)
    return columns2keep_zz, columns2keep_xxxyyxyy, columns2keep_xx


def keep_classical_or_quantum_measured_obs(X, choice:str, N:int, nonlocalobs_dict=None):
    """
    Returns matrix with the measured observables of interest and the offset term, ready for IPC calculation.
    Args:
        X (array): full matrix of nonlocal observables.
        choice (str): can be either 'classical', 'quantum' or 'xx'. Determines whether we keep the all-z observable
                        pairs, all the pairs that do not involve z at all or all-x observable pairs, respectively.
        N (int): system size, necessary for the potential generation of nonlocalobs_dict.
        nonlocalobs_dict (dict): if given, it is not internally recalculated.
    """
    if nonlocalobs_dict is None:
        nonlocalobs_dict = generate_nonlocal_obs_dict(N)
    columns2keep_classical, columns2keep_quantum, columns2keep_xx = get_indices_segregated_measured_obs_matrix(nonlocalobs_dict)
    indices = [0]
    if choice == 'classical':
        indices.extend(columns2keep_classical)
    elif choice == 'quantum':
        indices.extend(columns2keep_quantum)
    elif choice == 'xx':
        indices.extend(columns2keep_xx)
        
    return X[:, indices]


def generate_local_obs(N):
    return [Sigma(0, x, N) for x in range(N)]


def generate_z_zz(N):
    O = []
    SigmaZ = [Sigma(0, x, N) for x in range(N)]
    for i in range(N):
        O.extend(SigmaZ[i] @ SigmaZ[j] for j in range(i))
    O.extend(SigmaZ)
    return O
    
def generate_x_xx(N):
    O = []
    SigmaX = [Sigma(1, x, N) for x in range(N)]
    for i in range(N):
        O.extend(SigmaX[i] @ SigmaX[j] for j in range(i))
    O.extend(SigmaX)
    return O

def generate_y_yy(N):
    O = []
    SigmaY = [Sigma(2, x, N) for x in range(N)]
    for i in range(N):
        O.extend(SigmaY[i] @ SigmaY[j] for j in range(i))
    O.extend(SigmaY)
    return O

def generate_all_locnonloc_obs(N):
    SigmaZ = [Sigma(0, x, N) for x in range(N)]
    SigmaX = [Sigma(1, x, N) for x in range(N)]
    SigmaY = [Sigma(2, x, N) for x in range(N)]
    single_paulis = [SigmaZ, SigmaX, SigmaY]
    O = []
    for i in combinations(np.arange(N*3), 2):
        s0, s1 = i
        spinnum0, spinnum1 = s0 % N, s1 % N
        c0, c1 = s0 // N, s1 // N
        if spinnum0 != spinnum1:
            obs = single_paulis[c0][spinnum0] @ single_paulis[c1][spinnum1]
            O.append(obs)
    O.extend(SigmaZ)
    O.extend(SigmaX)
    O.extend(SigmaY)
    return O