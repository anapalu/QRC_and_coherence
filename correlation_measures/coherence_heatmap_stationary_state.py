#!/usr/bin/env python

import argparse
import os
from timeit import default_timer as timer

import numpy as np
from numba import complex128
from qrccoherence.evolution import C_l1, get_rho1_of_sk, initial_state
from qrccoherence.njitfuncts import fast_diag, fast_dot, fast_eigvals_eigvects, fast_kron
from qrccoherence.pauli_matrices import Ham

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"




parser = argparse.ArgumentParser(description="Run simulation of 2D plot of coherence in the stationary regime for \
                                            varying disorder amplitude W and global local field contribution h_mean.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input_injection", nargs="?", const='mixed_z', type=str, help="Input injection map. Can be \
                    mixed_z (default if --input_map is called but no value is assigned, standard Nakajima encoding), \
                        pure_zx (pure state in xz plane centered around z axis), mixed_x (Nakajima encodin along x) or \
                            purex (pure state in xz plane centered around x axis). Defaults to 'mixed_z'.")
parser.add_argument("-N", "--system_size", nargs="?", const=5, type=int, help="number of spins in the network.")
parser.add_argument("-Dt", "--time_between_inputs", nargs="?", const=10, type=float, help="time between input \
                                                                                            injections.")
parser.add_argument("-L", "--input_length_averaging", nargs="?", const=1000, type=int, help="number of input steps to \
                                                                    average over after wash-out.")
parser.add_argument("--Nsys_averaging", nargs="?", const=100, type=int, help="number of random realisations of the \
                                                                            system to average over.")
parser.add_argument("--wash_out", nargs="?", const=1000, type=int, help="number of input steps that are considered \
                                                                        enough to erase the reservoir's initial \
                                                                        condition and enter the stationary regime.")
args = parser.parse_args()
config = vars(args)

  
def main():    
    
    N = config['system_size']
    Dt = config['time_between_inputs']
    Js = 1
    wu_steps = config['wash_out']
    L = config['input_length_averaging']
    N_sys = config['Nsys_averaging']
    
    input_injection = config['input_injection']

    Ws = np.logspace(np.log10(0.01), np.log10(100), 40)
    hs = np.logspace(np.log10(0.01), np.log10(100), 40)

    initialrho_maxINcoherent = initial_state(N, initialisation='allfacingup')

    np.random.seed(5762)
    s_wu = np.random.rand(wu_steps)
    s_t = np.random.rand(L)

    np.random.seed(532)
    jotas = np.random.rand(N_sys, N, N)
    haches = np.random.rand(N_sys, N)


    ##WHERE WE WANT TO STORE OUR DATA
    file_path = f'./data/{input_injection}/coherence_heatmap/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = f'HMlowerr_N{N}_Dt{Dt}_L{L}_Nsys{N_sys}_lenW{len(Ws)}_lenh{len(hs)}.txt'
    fM = open(file_path + file_name, 'w+')
    fM.write('#Stationary coherence HM, same N_sys systems for every gamma and same input \
             string for all of them, every time. All systems initialised with all spins up. Parameters: \
             Dt={}, N={}, wu_steps={}, L={}, N_sys={}\n#Rows correspond to systems, \
             columns to (input) time step, and the (N_sys, L) matrix of data for the len(hs) points are \
             stacked one after the other, then these structures are stacked as well for the different \
             gammas separated by a line starting with tab+hashtag\n'.format(Dt, N, wu_steps, L, N_sys))
    

    total_time0 = timer()

    
    for j, W in enumerate(Ws):
        fM.write('####W = {}\n'.format(W))

        for jjj, h_mean in enumerate(hs):
            fM.write('#h = {}\n'.format(h_mean))
            for kk in range(N_sys):

                J = Js * 0.5 * (jotas[kk] + jotas[kk].T - 1)
                h = h_mean + W * haches[kk]
                
                 
                H = Ham(J, h, N)
                eigvals, P = fast_eigvals_eigvects(H)
                evol_op = fast_dot(P, fast_dot( fast_diag( np.exp(-1j * eigvals * Dt) ) , P.T))

                rho = initialrho_maxINcoherent

                for s in s_wu:
                    
                    ##Build rho1 from input
                    rho1 = get_rho1_of_sk(s, input_injection)

                    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
                    reduced_rho = np.einsum('ijik->jk', reshaped_rho)

                    rho = fast_kron(rho1, reduced_rho)

                    rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve


                for k, s in enumerate(s_t):
                    psi_sk = np.sqrt(1-s) * np.array((1,0)) + np.sqrt(s) * np.array((0,1))
                    rho1 = complex128(fast_kron(psi_sk, psi_sk.T).reshape(2,2))##Build rho1 for inputs

                    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
                    reduced_rho = np.einsum('ijik->jk', reshaped_rho)

                    rho = fast_kron(rho1, reduced_rho)

                    rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve

                    ### MEASUREMENT (right before feeding the next input)
                    fM.write(str(C_l1(rho)) + '\t')
                fM.write('\n')


    total_time1 = timer()
    fM.write('#elapsed time: ' + str(total_time1 - total_time0) )

    fM.close()
    


    
    
if __name__ == "__main__":
    main()