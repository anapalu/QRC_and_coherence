import argparse
import os

import numpy as np
from qrccoherence.evolution import function_factory, monitor_information_content
from qrccoherence.pauli_matrices import Sx, Sz

# from qrccoherence.saveload_data import retrieve_fixed_randomness ## FOR SOME REASON, THE RETRIEVED
## VECTORS WERE GIVING ME WEIRD DATA TYPE ERRORS IF I INPORTED THE FUNCTION

parser = argparse.ArgumentParser(description="Calculate correlation measures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("W", help="disorder strength of the local fields")
parser.add_argument("--input_injection", nargs="?", const='mixed_z', type=str, help="Input injection map. Can be \
                    mixed_z (default if --input_map is called but no value is assigned, standard Nakajima encoding), \
                        pure_zx (pure state in xz plane centered around z axis), mixed_x (Nakajima encodin along x) or \
                            purex (pure state in xz plane centered around x axis). Defaults to 'mixed_z'.")
args = parser.parse_args()
config = vars(args)


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



W = int(config['W'])
N = 5
input_injection = config['input_injection']#'purex'#'pure_zx'
if input_injection is None:
    input_injection = 'mixed_z'

file_path = f'./correlation_measures/data/{input_injection}/W{W}/'
#W{W}correlation_measures_N5_fixedsystems_shortperrlist_{input_map}'

Nsys = 10
# 0.001, 0.005, 0.01, 0.02, 0.05
p_err_list = [0.005, 0.05] 

Dt = 10
wu_steps = 500
L = 100

z0s = np.zeros((3, 2, len(p_err_list), Nsys)) ##(unitary, bitflip, phaseflip), (mean, var), (perr), (nsys)
T, CL, C, M, K, CM = z0s, z0s, z0s, z0s, z0s, z0s

lhs_arr = [T, CL, C, M, K, CM]

extended_perr = np.append(0, p_err_list)
intermediate_mat = np.zeros((len(lhs_arr), 2, 2, len(p_err_list) + 1, Nsys ))

rand_Jdisorder, rand_hdisorder, s_t, s_uk = retrieve_fixed_randomness()
s_wu = s_t[:wu_steps]
s_uk = s_t[wu_steps:L+wu_steps]

for n in range(Nsys):
    J = rand_Jdisorder[n]
    hdis = rand_hdisorder[n]
    h = 1 * np.ones(N) + W * hdis
    
    ## p = 0
    t, cL, c, m, _, _, _, _ = monitor_information_content(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=None, 
                                                            dephasteps=50, initialisation='maxcoh',
                                                            input_map = input_injection)
    rhs_arr = [t, cL, c, m, m-c, c-cL]
    for i, aa in enumerate(rhs_arr):
        intermediate_mat[i, 0, 0, 0, n] = intermediate_mat[i, 1, 0, 0, n] = np.mean(aa)
        intermediate_mat[i, 0, 1, 0, n] = intermediate_mat[i, 1, 1, 0, n] = np.var(aa)
    
    for j, p in enumerate(p_err_list):
        ## bit flip
        decoh = function_factory(p, Sx, N)

        t, cL, c, m, _, _, _, _ = monitor_information_content(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=decoh, 
                                                            dephasteps=50, initialisation='maxcoh',
                                                            input_map = input_injection)
        rhs_arr = [t, cL, c, m, m-c, c-cL]
        for i, aa in enumerate(rhs_arr):
            intermediate_mat[i, 0, 0, j+1, n] = np.mean(aa)
            intermediate_mat[i, 0, 1, j+1, n] = np.var(aa)

        ## phase flip
        decoh = function_factory(p, Sz, N)
        t, cL, c, m, _, _, _, _ = monitor_information_content(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=decoh, 
                                                            dephasteps=50, initialisation='maxcoh',
                                                            input_map = input_injection)
        rhs_arr = [t, cL, c, m, m-c, c-cL] #T, C_L, C, M, K, C_M
        for i, aa in enumerate(rhs_arr):
            intermediate_mat[i, 1, 0, j+1, n] = np.mean(aa)
            intermediate_mat[i, 1, 1, j+1, n] = np.var(aa)



final_mat = np.zeros((len(lhs_arr), 2, 2, len(p_err_list) + 1 ))
final_mat[:, :, 0, :] = np.mean(intermediate_mat[:, :, 0, :, :], axis = 3)
final_mat[:, :, 1, :] = np.sqrt(np.sum((intermediate_mat[:, :, 1, :, :])**2, axis = 3) ) / Nsys

file_path = f'./data/{input_injection}/W{W}/T_Cl_C_M_K_Cm/'
if not os.path.exists(file_path):
    os.makedirs(file_path)
file_name = 'correlations'

np.savez(file_path + file_name, final_mat=final_mat, s_wu=s_wu, s_uk=s_uk, p_errs=extended_perr)