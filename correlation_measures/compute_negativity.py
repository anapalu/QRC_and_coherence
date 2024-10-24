#!/usr/bin/env python

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import os
from timeit import default_timer as timer

import numpy as np
from qrccoherence.evolution import function_factory, monitor_negativity
from qrccoherence.pauli_matrices import Sx, Sz
from qrccoherence.saveload_data import retrieve_fixed_randomness


def list_of_floats(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser(description="Calculate average negativity over all possible partitions during \
                                    stationary regime", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('Ws', type=list_of_floats, help = "list containing the disorder amplitudes W to compute for each \
                                                        system, e.g., 0,10 .")
parser.add_argument('p_err_list', type=list_of_floats, help = "list containing the error probabilities to compute, e.g., \
                                                        0.001,0.005,0.01,0.02,0.05 . Both bit flip and phase flip noise \
                                                        are computed.")
# parser.add_argument("nsys", type=int, help="system index")
parser.add_argument("--input_injection", nargs="?", const='mixed_z', type=str, help="Input injection map. Can be \
                    mixed_z (default if --input_map is called but no value is assigned, standard Fujii encoding), \
                        pure_zx (pure state in xz plane centered around z axis), mixed_x (Fujii encoding along x) or \
                            purex (pure state in xz plane centered around x axis). Defaults to None, which will \
                                implement mixed_z.")
parser.add_argument("-N", "--system_size", nargs='?', const=5, type=int, help="size of the all-to-all connected spin \
                                                                                network. Defaults to 5.")

args = parser.parse_args()
config = vars(args)



def main():
    N = config['system_size']
    Dt = 10
    wu_steps = 500; L = 100
    Nsys = 10
    # n = config['nsys']
    input_injection = config['input_injection']
        
    Ws = config['Ws']
    perr_list = config['p_err_list']
    extended_perr_list = [0]
    extended_perr_list.extend(perr_list)
    
    for n in range(Nsys):
        
        for W in Ws:
            
            neg = np.zeros((len(extended_perr_list), 2, L)) #(perr, (bitflip, phaseflip), L)
            neg_std = np.zeros((len(extended_perr_list), 2, L)) #(perr, (bitflip, phaseflip), L)
        
        
            fpath_save = f'./correlation_measures/data/{input_injection}/W{W}/negativity/'
            if not os.path.exists(fpath_save):
                os.makedirs(fpath_save)
                
            file_name = 'sys{n}'
            
            rand_Jdisorder, rand_hdisorder, s_t, s_uk = retrieve_fixed_randomness()
            s_wu = s_t[:wu_steps]
            s_uk = s_t[wu_steps:L+wu_steps]
            
            J = rand_Jdisorder[n]
            hdis = rand_hdisorder[n]
            h = 1 * np.ones(N) + W * hdis
            neg, neg_std, _, _ = monitor_negativity(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=None, 
                                                    dephasteps=50, initialisation='maxcoh', input_map=input_injection)
            
            ## add p=0 data to both Sx and Sz columns
            neg[0, 0] += neg
            neg[0, 1] += neg
            neg_std[0, 0] += neg_std
            neg_std[0, 1] += neg_std
            
            ## Calculate noisy points
            for pidx, p_err in enumerate(perr_list):
                decoh_Sx = function_factory(p_err, Sx, N)
                decoh_Sz = function_factory(p_err, Sz, N)

                negx, neg_stdx, _, _ = monitor_negativity(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=decoh_Sx, 
                                                                dephasteps=50, initialisation='maxcoh', input_map=input_injection)
                negz, neg_stdz, _, _ = monitor_negativity(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=decoh_Sz, 
                                                                dephasteps=50, initialisation='maxcoh', input_map=input_injection)
                neg[pidx + 1, 0] += negx
                neg[pidx + 1, 1] += negz
                neg_std[pidx + 1, 0] += neg_stdx
                neg_std[pidx + 1, 1] += neg_stdz
            

            np.savez(fpath_save + file_name, extended_perr_list=extended_perr_list, 
                    coords=['x', 'z'], negativity=neg, negativity_standard_dev=neg_std)
        ## TODO: I CORRECTED THE TYPO HERE, SO WE NEED TO UPDATE DATA




if __name__ == "__main__":
    main()