#!/usr/bin/env python

import argparse
import os
from timeit import default_timer as timer

import numpy as np
from numba import complex128
from qrccoherence.evolution import evolve_system, function_factory, get_rho1_of_sk
from qrccoherence.pauli_matrices import (
    Ham,
    Sx,
    Sz,
    fast_diag,
    fast_dot,
    fast_eigvals_eigvects,
    fast_kron,
    fast_trace,
    generate_all_locnonloc_obs,
    generate_nonlocal_obs,
    generate_z_zz,
)
from qrccoherence.saveload_data import retrieve_fixed_randomness

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"



parser = argparse.ArgumentParser(description="Measure observables for IPC.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("W", type=int, help="disorder strength of the local fields")
parser.add_argument("p_err", type=float, help="error probability for decoherence strength")
parser.add_argument("coord", type=str, help="direction in which noise is introduced, must be either 'x' or 'z'.\
                                    For no noise, set p_err to 0 and the coordinate introduced will be ignored.")
parser.add_argument("nsys", type=int, help="system to calculate.")
parser.add_argument("obs", type=str, help="set of observables to measure along the QRC simulation.\
                                    Currently available options are 'classical_locnonloc' for retrieving only all local \
                                        Zs and all ZZ-pairs, 'classical_loc' for only all local Zs, 'all_nonloc' for all \
                                            possible 2-local observables and 'all_locnonloc', for all possible 2-local \
                                                observables plus all local observables. Provisionally added 'multiplex \
                                                    to account for time multiplexing of ZZ set with V=4.")
parser.add_argument("-i", "--input_injection", nargs="?", const='mixed_z', type=str, help="string defining the type of \
                                                                input injection. Currently accepted options are \
                                                                'mixed_x', 'mixed_z', 'pure_x' and 'pure_z'. Defaults to \
                                                                'mixed_z'.")
parser.add_argument("-N", "--system_size", nargs='?', const=5, type=int, help="size of the all-to-all connected spin \
                                                                                network. Defaults to 5.")
parser.add_argument("-V", "--time_multiplexing", nargs='?', const=1, type=int, help="number of measurements to take \
                                                                            between input injections. Currently not \
                                                                            implemented in a general manner, so defaults \
                                                                            to 1 on the inside and overwritten to 4 if\
                                                                            obs='multiplex'.")
args = parser.parse_args()
config = vars(args)


############INSERT MAIN
def main():

    W = config['W']
    
    N = config['system_size']
    Nsys = 10 ## even though we're not currently using it explicity, we need to pass it to the function 
                ## that retrieves the instances
    seed = 1234
    L = 10**5 
    rand_Jdisorder, rand_hdisorder, s_t, s_uk = retrieve_fixed_randomness(seed=seed, L=L, N=N, Nsys=Nsys)

    input_injection = config['input_injection']
        
    observables_to_measure = config['obs']
    if observables_to_measure == 'classical_locnonloc':
        O = generate_z_zz(N=N)
    elif observables_to_measure == 'classical_loc':
        O = generate_z_zz(N=N)[-N:]
    elif observables_to_measure == 'all_nonloc':
        O = generate_nonlocal_obs(N=N)
    elif observables_to_measure == 'all_locnonloc':
        O = generate_all_locnonloc_obs(N)
    
    ## NOT GENERAL, specific to our work (a shortcut, really, to avoid modifying the function to get the observables much)
    elif observables_to_measure == 'multiplex':
        O = generate_z_zz(N=N)[-N:]
        V = 4
        
    Dt = 10
    wu_steps = 1000 ## warm-up/wash-out steps
    s_wu = np.random.rand(wu_steps) ## inputs for warm-up/wash-out
    dephasteps = 50
    
    V = config['time_multiplexing']
        
    coord = config['coord']
    if coord == 'x': #Sz for phase flip and Sx for bit flip
        Szx = Sx
        Szx_str = 'Sx'
    elif coord == 'z':
        Szx = Sz
        Szx_str = 'Sz'
        
    p = config['p_err']
    if p == 0:
        Szx = Sz ### doesn't matter, we won't use it. We just need to set it so that it passes through.
        Szx_str = 'ideal'
        
    folder_X = f'./data/{input_injection}/W{W}/{Szx_str}/{p}/'
    # Check whether the specified path exists or not and create it if it doesn't
    if not os.path.exists(folder_X):
        os.makedirs(folder_X)
    fname_X = f'X_{observables_to_measure}' 
    
    check_if_multiplex = False
    if V != 1:
        check_if_multiplex = True
        # fname_X = f'X_{observables_to_measure}_V{V}' ## A more general implementation should have something like this
    

    decoh = function_factory(p, Szx, N)
    
    check_if_unitary = False
    dt = Dt/dephasteps
    if p == 0:
        dt = Dt
        dephasteps = 0
        check_if_unitary = True
            
    # for nsys in range(Nsys):
    nsys = config['nsys']
    simulation_params = f'N={N}, nsys={nsys}, Dt={Dt}, wu_steps={wu_steps}, dephasteps={dephasteps}, \
        Nsys={Nsys}, seed={seed}, L={L}, h_mean=1, V={V} (time multiplexing)'
    
    
    t0 = timer()

    ## build instance
    J = rand_Jdisorder[nsys]
    hdis = rand_hdisorder[nsys]
    h = 1 * np.ones(N) + W * hdis
    H = Ham(J, h, N)
    
    ## initialise state. We start from the maximally coherent state, but this is irrelevant for QRC performance.
    rho = complex128(0.5**N * np.ones((2**N, 2**N)) )
    eigvals, P = fast_eigvals_eigvects(H)
    
    multiplex_indices = [int(x) for x in np.round((dephasteps - 1) * np.arange(1, V+1) / V)][:-1] ## we remove the last
                                                            ## point because we want to measure it in any case
                                                            
    evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * dt) ), np.conj(P).T))
    for s in s_wu:
        ##Building rho1
        rho1 = get_rho1_of_sk(s, input_injection)

        ##Taking the partial trace
        reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
        reduced_rho = np.einsum('ijik->jk', reshaped_rho)

        ##Feed input
        rho = fast_kron(rho1, reduced_rho)
        
        ##Evolve system, introduce decoherence if necessary
        rho = evolve_system(rho, evol_op, decoh, dephasteps, check_if_unitary)
        
    ## get measurement data
    X = np.empty((len(s_t) + len(s_uk), len(O)+1))
    for i, s in enumerate(np.append(s_t, s_uk)):
        ##Building rho1
        rho1 = get_rho1_of_sk(s, input_injection)

        ##Taking the partial trace
        reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
        reduced_rho = np.einsum('ijik->jk', reshaped_rho)

        ##Feed input
        rho = fast_kron(rho1, reduced_rho)
        
        ##Evolve system, introduce decoherence if necessary
        o = np.empty(len(O) * V)
        v = 0 ## index for adding observables in an orderly manner
        if check_if_unitary:
            rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve unitarily
        else:
            for k in range(dephasteps):
                # first step: unitary evolution
                rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve

                # second step: dephasing
                rho = decoh(rho)
                rho = rho/np.trace(rho)
                if check_if_multiplex and (k == multiplex_indices).any():
                    for kkk in range(len(O)):
                        o[len(O)*v + kkk] = fast_trace(np.real(fast_dot(O[kkk], rho))) ##multiplexed measurement
                    v += 1
        
        for kkk in range(len(O)):
            o[len(O)*v + kkk] = fast_trace(np.real(fast_dot(O[kkk], rho))) ##measurement

        X[i] = np.append(1, o ) ## the first column is all ones to account for the weight going with a constant.
  
    t1 = timer()
    sim_time = t1 - t0
    np.savez(folder_X  + fname_X + f'_nsys{nsys}.npz', X=X, joint_simulation_time=sim_time,
                simulation_params=simulation_params)


    
if __name__ == "__main__":
    main()
