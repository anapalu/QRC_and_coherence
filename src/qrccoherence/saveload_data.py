import os

import numpy as np
from qrccoherence.pauli_matrices import generate_nonlocal_obs_dict, get_indices_segregated_measured_obs_matrix


## Fix randomness reservoirs
def retrieve_fixed_randomness(seed=1234, N=5, Nsys=10, L=10**5):
    np.random.seed(seed)
    rand_Jdisorder = []
    rand_hdisorder = []
    for _ in range(Nsys):
        rand_Jdisorder.append(np.random.rand(2**N, 2**N))
        rand_hdisorder.append(np.random.rand(N))
    rand_st = np.random.rand(L)
    rand_suk = np.copy(rand_st)
    return rand_Jdisorder, rand_hdisorder, rand_st, rand_suk


def save_data(filename, variable_names, variables): ### we store each array in a line of the .txt
    with open(filename, 'w+') as f: ## using with ensures closure of the file at the end
        f.write('#')
        for var_names in variable_names:
            f.write(str(var_names) +'\t')
        f.write('\n')
    with open(filename, 'a+') as f:
        for var in variables:
            for i in var:
                f.write(str(i))
                f.write('\t')
            if any(x!=y for x,y in zip(var, variables[-1])):
                f.write('\n')


    return print(f'Data stored in {filename}')
    
    
def retrieve_data(filename): 
    with open(filename, 'r') as f:
        contents = f.readlines()
        data = []
        for line in contents:
            if line[0] != '#':
                dat = [float(x) for x in line.split('\t')[:-1]]
                data.append(np.asarray(dat))
    return np.asarray(data)

def get_path2save_ipc(W:float, Szx:str, perr, ipc_flag:str, nsys:int, D:int, 
                  input_injection:str='mixed_z', gaussian=None, add4mock=None, V=1) -> str:
    """
    Returns the corresponding path + filename with the structure we'll use to retrieve it for plotting. If the 
    corresponding directory didn't exist already, it creates it.
    Arguments:
        W (float): amplitude of disorder in the local fields.
        Szx (str): coordinate direction in which noise is introduced, must be either 'Sx', 'Sz' or 'ideal'.
        perr (float|int): error probability, characterising noise strength.
        ipc_flag (str): label indicating the set of observables to use. Currently supported labels are 'classical_loc'
                        (Z set), 'classical_nonloc' (ZZ set), 'classical_locnonloc' (Z+ZZ set), 'multiplex' (multiplexed
                        ZZ set) and 'quantum_nonloc' (XX + XY + YX + YY set).
        nsys (int): system index.
        D (int): degree of the contributions to the IPC.
        input_injection (str, optional): label specifying input injection scheme. Currently supported labels are 
                                        'mixed_z', mixed_x', 'pure_x' and 'pure_z'.
        gaussian (None|float, optional): standard deviation of the gaussian noise to be added to the measured observables obtained
                                by simulation in order to consider finite sample sizes. The relation of this parameter
                                to sample size is gaussian = \sigma = \sqrt{1/N_{obs}}. If None, no gaussian noise is
                                applied to the observables. Defaults to None. 
        add4mock (None|str, optional): if not None, appends the given string to the end of the file name (more 
                                        precisely, it appends '_' + add4mmock). Useful for testing. Defaults to None.
        V (int, optional): time multiplexing parameter, characterising the number of measurements taken between input
                            injections. Defaults to 1.
        
    Returns:
        (str): file path + name where the IPC contributions will be stored.
    """
    if gaussian is None:
        gaussian = 0
    path2save = f'./data/{input_injection}/W{W}/{ipc_flag}/{Szx}/{perr}/g{gaussian}/'
    
    deg_flag = f'_deg{D}'
    fileend = f'_nsys{nsys}'
    extension = '.npz'
    if V != 1:
        fileend += f'_V{V}'
    if add4mock is not None:
        fileend += f'_{add4mock}'
    
    filename2save = 'cc'
    
    # Check whether the specified path exists or not and create it if it doesn't
    if not os.path.exists(path2save):
        os.makedirs(path2save)

    return path2save + filename2save + deg_flag + fileend + extension

def get_relevant_measured_obs(W, Szx, perr, ipc_flag, nsys, N, gaussian, input_injection, V=1):
    """
    Returns observable matrix X ready for IPC calculation. New observable sets should be defined here as well.
    
    Arguments:
        W (float): amplitude of disorder in the local fields.
        Szx (str): coordinate direction in which noise is introduced, must be either 'Sx', 'Sz' or 'ideal'.
        perr (float): error probability, characterising noise strength.
        ipc_flag (str): label indicating the set of observables to use. Currently supported labels here are 
                        'classical_loc' (Z set), 'classical_nonloc' (ZZ set), 'classical_locnonloc' (Z+ZZ set), 
                        'quantum_nonloc' (XX + XY + YX + YY set), 'all_nonloc' (all possible two-body terms) and  
                        'all_locnonloc' (all possible two-body terms + all possible local terms).
        nsys (int): system index from the instance set.
        N (int): system size.
        gaussian (None|float): standard deviation of the gaussian noise to apply affecting the measurements.
        input_injection (str): input injection scheme. Currently accepted options are 'mixed_x', 'mixed_z', 'pure_x'
                                and 'pure_z'.
        V (int): time multiplexing parameter, indicates how many measurements are taken before next input injection. \
                    Measurements are always equally spaced along the time interval between input injections Dt. \
                        Defaults to 1 (no time multiplexing, measure only right before next input injection).
                                
    Returns:
        X (np.array): matrix of output functions ready for IPC calculation.
    """
    path2retrieve = f'../measure_obs/data/{input_injection}/W{W}/{Szx}/{perr}/'
    
    indices_zz, indices_xxxyyxyy, indices_xx = get_indices_segregated_measured_obs_matrix(generate_nonlocal_obs_dict(N))
    
    if ipc_flag == 'classical_loc':
        if os.path.exists(path2retrieve + f'X_{ipc_flag}_{nsys}.npz'):
            filename2retrieve = f'X_{ipc_flag}_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X']
        elif os.path.exists(path2retrieve + f'X_classical_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_classical_locnonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            idx2retrieve = [0]
            num_z_pairs = int(N * (N-1) // 2)
            idx_obs = list(range(num_z_pairs, num_z_pairs + N))
            idx2retrieve.extend(idx_obs)
            X = f['X'][:, idx2retrieve]
        elif os.path.exists(path2retrieve + f'X_all_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_locnonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            idx2retrieve = [0]
            idx_obs = list(range(-3*N, -2*N))
            idx2retrieve.extend(idx_obs)
            X = f['X'][:, idx2retrieve]
            
    elif ipc_flag == 'classical_nonloc':
        idx2retrieve = [0]
        idx2retrieve.extend(indices_zz)
        if os.path.exists(path2retrieve + f'X_classical_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_classical_locnonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            num_z_pairs = int(N * (N-1) // 2)
            X = f['X'][:, :num_z_pairs + 1]
        elif os.path.exists(path2retrieve + f'X_all_nonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_nonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X'][:, idx2retrieve]
        elif os.path.exists(path2retrieve + f'X_all_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_locnonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X'][:, idx2retrieve]
        
    elif ipc_flag == 'classical_locnonloc': 
        if os.path.exists(path2retrieve + f'X_{ipc_flag}_{nsys}.npz'):
            filename2retrieve = f'X_{ipc_flag}_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X']
        elif os.path.exists(path2retrieve + f'X_all_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_locnonloc_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            idx2retrieve = [0]
            idx2retrieve.extend(indices_zz)
            idx_loc = list(range(-3*N, -2*N))
            idx2retrieve.extend(idx_loc)
            X = f['X'][:, idx2retrieve]
            
    elif ipc_flag == 'quantum_nonloc':
        idx2retrieve = [0]
        idx2retrieve.extend(indices_xxxyyxyy)
        if os.path.exists(path2retrieve + f'X_all_nonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_nonloc_{nsys}.npz'
        elif os.path.exists(path2retrieve + f'X_all_locnonloc_{nsys}.npz'):
            filename2retrieve = f'X_all_locnonloc_{nsys}.npz'
        f = np.load(path2retrieve + filename2retrieve)
        X = f['X'][:, idx2retrieve]
        
    elif ipc_flag == 'all_locnonloc': 
        if os.path.exists(path2retrieve + f'X_{ipc_flag}_{nsys}.npz'):
            filename2retrieve = f'X_{ipc_flag}_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X']
    elif ipc_flag == 'multiplex':
        if os.path.exists(path2retrieve + f'X_{ipc_flag}_{nsys}.npz'):
            filename2retrieve = f'X_{ipc_flag}_{nsys}.npz'
            f = np.load(path2retrieve + filename2retrieve)
            X = f['X']
    
    
    if gaussian is not None:
        shapeX0, shapeX1 = np.shape(X)
        X[:, 1:] += np.random.normal(0, gaussian, (shapeX0, shapeX1 - 1))
    return X