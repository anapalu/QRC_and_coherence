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
from qrccoherence.ipc import degD_ipc
from qrccoherence.pauli_matrices import (
    generate_nonlocal_obs_dict,
    get_indices_segregated_measured_obs_matrix,
    keep_classical_or_quantum_measured_obs,
)
from qrccoherence.saveload_data import get_path2save_ipc, get_relevant_measured_obs, retrieve_fixed_randomness


# Define a custom argument type for a list of integers
def list_of_floats(arg):
    return list(map(float, arg.split(',')))
def list_of_ints(arg):
    return list(map(int, arg.split(',')))
 

parser = argparse.ArgumentParser(description="Calculate IPC",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("W", help="disorder strength of the local fields")
parser.add_argument("p_err", type=float, help="error probability for decoherence strength")
parser.add_argument("ipc_flag", type=str,
                    help="measured observables to consider for the measurement matrix. \
                        Can be 'all_nonloc', 'classical_nonloc', 'classical_locnonloc', 'quantum_nonloc',\
                            'classical_loc' or 'all_locnonloc' ('multiplex' IS IMPLEMENTED IN A DIFFERENT .py)")
parser.add_argument("coord", type=str, help="direction in which noise is introduced, must be either 'x' or 'z'.\
                                    For no noise, set p_err to 0 and the coordinate introduced will be ignored.")
parser.add_argument('max_delays', type=list_of_ints, help = "list containing the maximum delay to compute for each \
                                                                degrees (its length has to be as least the number of \
                                                                    degrees you want to compute). For the ergodic \
                                                                        region (and N=5), a good guess is \
                                                                            30,30,30,30,10,10 , while in for MBL the \
                                                                                linear memory may go up to 300.")
parser.add_argument('thresholds', type=list_of_floats, help = "list containing the thresholds to stop trying to get \
                                                                contributions out of each degree (its length has to be \
                                                                as least the number of degrees you want to compute). We \
                                                                allow to go below this threshold cc0max times before \
                                                                switching to the next degree. For the ergodic region \
                                                                (and N=5) the ergodic region (and N=5), a good guess is \
                                                                4e-4,4e-4,5e-4,5e-4,6e-4,7e-4 .")
parser.add_argument("-i", "--input_injection", type=str, help="string defining the type of input injection. Currently \
                                                                accepted options are 'mixed_x', 'mixed_z', 'pure_x' \
                                                                    and 'pure_z'. Dafaults to None, which will \
                                                                        implement 'mixed_z'.") #TODO: FINISH IMPLEMENTING THIS
parser.add_argument("--add4mock", help="end part of storage file to be included in order to identify \
                        tests of this script")
parser.add_argument("-g", "--gaussian", type=float, help="standard deviation of the gaussian noise affecting the \
                                        measurements of the observables. Defaults to None.")
parser.add_argument("--Dmin", type=int, help="minimum degree to consider. Defaults \
                                        to 1.")
parser.add_argument("--Dmax", type=int, help="maximum degree to consider. Defaults \
                                        to 6.")
parser.add_argument("--Dstep", type=int, help="step on array of degrees to \
                            consider. Set to 2 in order to look into odd contributions \
                                exclusively. Defaults to 1.")
parser.add_argument("--cc0max", type=int, help="number of consecutive degrees that can \
                                return a contribution below threshold before terminating the calculation of the \
                                    IPC. Defaults to 2.")
args = parser.parse_args()
config = vars(args)
    


def main():
    W = config['W']

    Nsys = 10
    _, _, st, suk = retrieve_fixed_randomness()
    N = config['system_size']
    Dmin = 1
    Dmax = 6
    Dstep = 1
    if config['Dmax']  is not None:
        Dmax = config['Dmax'] 
    if config['Dmin']  is not None:
        Dmin = config['Dmin'] 
    if config['Dstep'] is not None:
        Dstep = config['Dstep']
    
    perr = config['p_err'] 
    
    coord = config['coord']
    if coord == 'x':  # Sz for phase flip and Sx for bit flip
        Szx = 'Sx'
    elif coord == 'z':
        Szx = 'Sz'
       
    
    if np.round(perr, 6) == 0:  ## NOW AN ERROR PROBABILITY OF 0 OVERRIDES ANY COORD SPECIFICATION
        Szx = 'ideal'
        perr = int(perr)
        
    ipc_flag = config['observables']  
    
    input_injection = config['input_injection']
    if input_injection is None:
        input_injection = 'mixed_z'
    add4mock=config['add4mock']
    
    
    max_counts_cc0max = 2
    if config['cc0max'] is not None:
        max_counts_cc0max = config['cc0max']

    max_delay = config['max_delays']
    threshold = config['thesholds']
    
    nsystems = np.arange(0, Nsys)
    degrees = np.arange(Dmin, Dmax+1)[::Dstep] # the +1 is so that Dmax is also calculated
    
    gaussian = config['gaussian']
    
    if gaussian is not None:
        info_addendum = f' and gaussian noise of std={gaussian}'
    else:
        info_addendum = ''

    info_message = f'collected capacitance string for the case W={W}, input_injection={input_injection}, noise={Szx},\
                     p_err={perr}, observables={ipc_flag}' + info_addendum + '\n'
    
    for nsys in nsystems:

        X = get_relevant_measured_obs(W, Szx, perr, ipc_flag, nsys, N, gaussian=gaussian, 
                                      input_injection=input_injection)
        
        counts_cc0max = 0
        max_found_delay = None

        for D in degrees:
            if max_found_delay is not None and max_found_delay != 0: ## without the second condition, if there's a 
                    ## 0 contribution in degree 1, it won't keep looking
                if max_found_delay < max_delay[D-1]:
                    max_delay[D-1] = max_found_delay
            
            t0 = timer()
          
            cc, max_found_delay = degD_ipc(D, X, st, suk, max_delay[D-1], threshold[D-1], max_counts=2, print_threshold=False)
           
            t1 = timer()
            sim_time = t1 - t0
         
            info_addendum = f'for D={D} and system nsys={nsys}. Stopping criteria were \
                            max_delay={max_delay[D-1]} and threshold={threshold[D-1]}. \
                                Simulation took {sim_time} seconds.'
            info = info_message + info_addendum
          
            fname = get_path2save_ipc(W, Szx, perr, ipc_flag, nsys, D, 
                                  add4mock=add4mock,
                                  gaussian=gaussian)
            np.savez(fname, C_deg=np.sum(cc), cc=cc,
                     threshold=threshold[D-1], max_delay=max_delay[D-1],
                     simulation_time=sim_time, info_message=info)
            
            if np.sum(cc) == 0:
                counts_cc0max += 1
            if counts_cc0max == max_counts_cc0max:
                break
   


if __name__ == "__main__":
    main()