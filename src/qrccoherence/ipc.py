from abc import ABC
from itertools import combinations, permutations
from timeit import default_timer as timer

import numpy as np
from qrccoherence.combinatorial_helper_functions import (
    admissible_orderings,
    combinations_ordered_by_window,
    get_all_decompositions,
    get_delayed_sk,
    get_perm_in_order_basis,
)
from qrccoherence.config import logger
from qrccoherence.njitfuncts import get_ipc_tiny
from qrccoherence.saveload_data import save_data
from scipy.special import legendre

pols = np.array([legendre(x) for x in range(30)], dtype='object')

# def get_ipc_tiny(X, y_t, y0):
#     L = len(y_t)
#     WW = la.lstsq(X[:L], y_t)[0].T
#     y = np.sum(WW * X[L:], axis = 1)

#     return 1-np.mean((y-y0)**2)/np.mean(y0**2)

def degD_ipc(D, X, st, suk, max_delay, threshold, max_counts=2, print_threshold=True): ## For studying contributions to different degrees in the search for the threshold
    """
    Calculate IPC contribution from a particular degree.
    """
    
    
    s_t = 2*st - 1
    s_uk = 2*suk - 1

    delayed_s_t = get_delayed_sk(s_t, max_delay) ## get the delayed versions of the input
    delayed_s_uk = get_delayed_sk(s_uk, max_delay)

    cc = []

    c_min = threshold
    max_delay_D = max_delay

    decs = get_all_decompositions(D) ## all possible decompositions of D

    found_max_delay_D = []

    for nv in range(D):

        decs_nvar = decs[nv] ## all possible decompositions of D in nv summands
        nv = nv + 1 ##now it's the actual number of variables we have
        
        combs = list(combinations(range(max_delay_D + 1), nv))
        
        if len(combs): ### proceed only if the list of combinations is not empty, which
            ## will eventually happen if the degree D is larger than the stipulated 
            ### maximum delay

            for pow_list in decs_nvar: ##pow_list IS A LIST CONTAINING THE DEGREES d_i
                
                ###WE MUST TREAT SEPARATELY THE CASE WITH len(pow_list) == 1 BECAUSE OF THE FUNCTION permutations
                if len(pow_list) == 1:
                    for positions in combs:

                        ############
                        y_t = pols[pow_list][0](delayed_s_t[positions])
                        y0 = pols[pow_list][0](delayed_s_uk[positions])

                        c = get_ipc_tiny(X, y_t, y0)
                        #############print('c ', c, '\n')
                        cc.append(c)
                        ############
                        if c <= c_min:  # if we hit the threshold
                            cc = cc[:-1]  # undo the addtion
                            found_max_delay_D.append(np.max(positions))
                            if print_threshold:
                                print('we hit threshold at', positions, np.max(positions))
                            break
                else:
                    number_distinct_minidegs = len(np.unique(pow_list))
                    w = 0 ## window in which we have failed, initially set to 0
                    counts = 0
                    some_contribution = 0

                    sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}
                    for positions in combinations_ordered_by_window(combs):
                        window = max(positions) - min(positions)

                        #change of windows after we have exhausted the current one
                        if counts >= max_counts and window != w:
                            counts = 0
                            if some_contribution == 0:
                                break
                            some_contribution = 0
                            sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}

    #######    WE HAVE TO INTRODUCE BY HAND THE CALCULATION OF THE FIRST ELEMENT OF A NEW WINDOW
                            if number_distinct_minidegs == 1:

                                ############
                                y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                                y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                                c = get_ipc_tiny(X, y_t, y0)
                                cc.append(c)

                                some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                                ############

                                if c <= c_min:
                                    w = window
                                    cc = cc[:-1] #undo the addtion
                                    counts += 1
                                    found_max_delay_D.append(np.max(positions))
                                    if print_threshold:
                                        print('we hit threshold at', positions, np.max(positions))
                                    some_contribution -= 1

                            else:  ###we need to iterate over permutations

                                count_total_perms = 0
                                failed_perms = 0

                                for rearranged_positions in admissible_orderings(pow_list, positions):
                                                            ##the resulting order already seems the most convenient
                                    perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) )
                                    if sparingperms_dict[perm_order_basis] == 1:
                                        failed_perms += 1
                                    else:

                                        rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                        ############
                                        y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                        y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                        c = get_ipc_tiny(X, y_t, y0)
                                        cc.append(c)
                                        some_contribution += 1
                                        ############
                                        if c <= c_min:
                                            sparingperms_dict[perm_order_basis] = 1
                                            w = window
                                            cc = cc[:-1] #undo the addtion
                                            failed_perms += 1
                                            found_max_delay_D.append(np.max(positions))
                                            if print_threshold:
                                                print('we hit threshold at', positions, np.max(positions))
                                            some_contribution -= 1
                                            #################print('we hit threshold, with c=', c)
                                    count_total_perms += 1
                                ## we want to consider that we have reached max_counts whenever there are no permutations
                                ## valid on those positions
                                if failed_perms == count_total_perms:
                                    counts = max_counts 
                                    w = window

                        elif counts >= max_counts:
                            pass
                        elif number_distinct_minidegs == 1:

                            ############
                            y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                            y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                            c = get_ipc_tiny(X, y_t, y0)
                            #################print('c ', c, '\n')
                            cc.append(c)
                            some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                            ############
                            if c <= c_min:
                                w = window
                                cc = cc[:-1] #undo the addtion
                                counts += 1
                                found_max_delay_D.append(np.max(positions))
                                if print_threshold:
                                    print('we hit threshold at', positions, np.max(positions))
                                some_contribution -= 1
                                #####################


                        else:  ###we need to iterate over permutations

                            count_total_perms = 0
                            failed_perms = 0

                            for rearranged_positions in admissible_orderings(pow_list, positions):
                                                    ##the resulting order already seems the most convenient
                                perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) )
                                if sparingperms_dict[perm_order_basis] == 1:
                                    failed_perms += 1
                                else:

                                    rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                    ############
                                    y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                    y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                    c = get_ipc_tiny(X, y_t, y0)
                                    cc.append(c)
                                    some_contribution += 1
                                    ############

                                    if c <= c_min:
                                        sparingperms_dict[perm_order_basis] = 1
                                        w = window
                                        cc = cc[:-1] #undo the addtion
                                        failed_perms += 1
                                        found_max_delay_D.append(np.max(positions))
                                        if print_threshold:
                                            print('we hit threshold at', positions, np.max(positions))
                                        some_contribution -= 1
                                        ##########################
                                count_total_perms += 1
                            ## we want to consider that we have reached max_counts whenever there are no permutations
                            ## valid on those positions
                            if failed_perms == count_total_perms:
                                counts = max_counts 
                                w = window
            if cc != [] and cc[-1] <= c_min:
                found_max_delay_D.append(np.max(positions))
                if print_threshold:
                    print('we hit threshold at', positions, np.max(positions))
                break
    if found_max_delay_D != []:
        if print_threshold:
            print('next max_delay should be', np.max(found_max_delay_D))
        return cc, np.max(found_max_delay_D)
    else:
        return cc, None


















def full_IPC_plus_cc_singlesys(fncc, X, st, suk,
                      max_delay, threshold, message_fncc=None,
                     max_counts=2, min_tot_deg=1, max_tot_deg=6):
    # sourcery skip: ensure-file-closed, hoist-statement-from-if, remove-redundant-if, remove-redundant-pass, remove-unnecessary-else
    """
    Calculation of the IPC.
    Args:
        fncc (str): path + name of the file to store the individual contributions
                    to the IPC of a single system.
        X (array): collected observables for fitting.
        st (array): training input as if was fed to the reservoir (i.e., between 0 and 1).
        suk (array): test input as if was fed to the reservoir (i.e., between 0 and 1).
        max_delay (list|array):contains maximum delays to consider when calculating each degree.
                                If the threshold is hit, the maximum delay for the next degree will
                                be dynamically set from the highest delay that hit the noise threshold for that degree.
        threshold (list|array): list containing the ordered lower bounds to consider a
                            contribution at a given capacity or discard it as noise.
        message_fncc (str): info to print as text in the first line of fncc.
        max_counts (int): number of times we allow to go below the threshold. Defaults to 2.
        min_tot_deg (int): minimum degree to look for contributions. Defaults to 1.
        max_tot_deg (int): maximum degree to look for contributions. Defaults to 6.
    Returns:
        C (list): capacities for each degree up to max_tot_deg.
        cc (list): individual contributions to capacity.
    """

    ## Initialise files
    ##WHERE WE WANT TO STORE OUR DATA
    filename3 = fncc
    fcc = open(filename3, 'w+')
    fcc.write(message_fncc) 
    ### WE REMOVED THE FINAL'\n' FROM HERE TO PUT IT AT THE BEGINNING OF DEGREE SWITCHING
    
    total_time0 = timer()

    s_t = 2*st - 1
    s_uk = 2*suk - 1

    delayed_s_t = get_delayed_sk(s_t, max_delay[0]) ## get the delayed versions of the input, 0th max delay should be the longest
    delayed_s_uk = get_delayed_sk(s_uk, max_delay[0])

    C = np.zeros(max_tot_deg)
    cc = []

    found_max_delay_D = []  ## variable to dynamically update the maximum delay we can expect in the next degree

    for D in range(min_tot_deg, max_tot_deg+1):
        fcc. write('\n')

        c_min = threshold[D-1]

        
        if found_max_delay_D != []: ##update max delay with previous degree threshold hit
            max_delay[D-1] = np.max(found_max_delay_D)
        
        max_delay_D = max_delay[D-1]

        decs = get_all_decompositions(D) ## all possible decompositions of D
        found_max_delay_D = []

        for nv in range(D):
            decs_nvar = decs[nv] ## all possible decompositions of D in nv summands
            nv = nv + 1 ##now it's the actual number of variables we have
            
            combs = list(combinations(range(max_delay_D + 1), nv)) 
            
            if len(combs): ### proceed only if the list of combinations is not empty, which
            ## will eventually happen if the degree D is larger than the stipulated 
            ### maximum delay
                
                for pow_list in decs_nvar: #range(len(decs_nvar)): ##pow_list IS A LIST CONTAINING THE DEGREES d_i
                    
                    ###WE MUST TREAT SEPARATELY THE CASE WITH len(pow_list) == 1 BECAUSE OF THE FUNCTION permutations
                    if len(pow_list) == 1:
                        for positions in combs:

                            ############
                            y_t = pols[pow_list][0](delayed_s_t[positions])
                            y0 = pols[pow_list][0](delayed_s_uk[positions])

                            c = get_ipc_tiny(X, y_t, y0)
                            #############print('c ', c, '\n')
                            cc.append(c)
                            
                            fcc.write(str(cc) + '\t')
                                
                            C[D-1] += c
                            ############
                            if c <= c_min: ## if we hit the threshold
                                cc = cc[:-1] #undo the addtion
                                found_max_delay_D.append(np.max(positions))
                                C[D-1] -= c
                                break
                    else:
                        number_distinct_minidegs = len(np.unique(pow_list))
                        w = 0 ## window in which we have failed, initially set to 0
                        counts = 0
                        some_contribution = 0

                        sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}
                                ## 0: calculate, 1: don't bother. Dictionary for optimization: avoids computing
                                # those permutations that proved to be unsufficient
                        for positions in combinations_ordered_by_window(combs):
                            window = max(positions) - min(positions)

                            #change of windows after we have exhausted the current one
                            if counts >= max_counts and window != w:
                                counts = 0
                                if some_contribution == 0: 
                                    break
                                else:
                                    some_contribution = 0
                                    sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}

    #######    WE HAVE TO INTRODUCE BY HAND THE CALCULATION OF THE FIRST ELEMENT OF A NEW WINDOW BY HAND
                                    if number_distinct_minidegs == 1:

                                        ############
                                        y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                                        y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                                        c = get_ipc_tiny(X, y_t, y0)
                                        cc.append(c)
                                        
                                        fcc.write(str(cc) + '\t')
                
                                        C[D-1] += c
                                        some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                                        ############

                                        if c <= c_min:
                                            w = window
                                            cc = cc[:-1] #undo the addtion
                                            C[D-1] -= c
                                            found_max_delay_D.append(np.max(positions))
                                            counts += 1
                                            some_contribution -= 1
                                            
                                    else:  ###we need to iterate over permutations

                                        count_total_perms = 0
                                        failed_perms = 0

                                        for rearranged_positions in admissible_orderings(pow_list, positions):
                                                                ##the resulting order already seems the most convenient
                                            perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) ) 
                                                ##it needs to be a tuple because we generated sparingperms_dict with permutations()

                                            if sparingperms_dict[perm_order_basis] == 1:
                                                count_total_perms += 1
                                                failed_perms += 1
                                                pass

                                            else:

                                                count_total_perms += 1
                                                rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                                ############
                                                y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                                y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                                c = get_ipc_tiny(X, y_t, y0)
                                                cc.append(c)
                                                
                                                fcc.write(str(cc) + '\t')

                                                C[D-1] += c
                                                some_contribution += 1
                                                ############
                                                if c <= c_min:
                                                    sparingperms_dict[perm_order_basis] = 1
                                                    w = window
                                                    cc = cc[:-1] #undo the addtion
                                                    C[D-1] -= c
                                                    found_max_delay_D.append(np.max(positions))
                                                    failed_perms += 1
                                                    some_contribution -= 1
                                                    #################print('we hit threshold, with c=', c)
                                        ## we want to consider that we have reached max_counts whenever there are no permutations
                                        ## valid on those positions
                                        if failed_perms == count_total_perms:
                                            counts = max_counts 
                                            w = window
                                            
        ####### FINAL DE PARCHE CHAPUCERO                                     


                            elif counts >= max_counts and window == w:
                                pass
                            ###go faster for the case when all minidegs are the same by just doing the first possible ordering
                            elif number_distinct_minidegs == 1:

                                ############
                                y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                                y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                                c = get_ipc_tiny(X, y_t, y0)
                                #################print('c ', c, '\n')
                                cc.append(c)
                                
                                fcc.write(str(cc) + '\t')

                                C[D-1] += c
                                some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                                ############
                                if c <= c_min:
                                    w = window
                                    cc = cc[:-1] #undo the addtion
                                    C[D-1] -= c
                                    found_max_delay_D.append(np.max(positions))
                                    counts += 1
                                    some_contribution -= 1
                                    #####################


                            else:  ###we need to iterate over permutations

                                count_total_perms = 0
                                failed_perms = 0

                                for rearranged_positions in admissible_orderings(pow_list, positions):
                                                        ##the resulting order already seems the most convenient
                                    perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) ) 
                                        ##it needs to be a tuple because we generated sparingperms_dict with permutations()

                                    if sparingperms_dict[perm_order_basis] == 1:
                                        count_total_perms += 1
                                        failed_perms += 1
                                        pass
                                    else:

                                        count_total_perms += 1
                                        rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                        ############
                                        y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                        y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                        c = get_ipc_tiny(X, y_t, y0)
                                        cc.append(c)
                                        
                                        fcc.write(str(cc) + '\t')

                                        C[D-1] += c
                                        some_contribution += 1
                                        ############

                                        if c <= c_min:
                                            sparingperms_dict[perm_order_basis] = 1
                                            w = window
                                            cc = cc[:-1] #undo the addtion
                                            C[D-1] -= c
                                            found_max_delay_D.append(np.max(positions))
                                            failed_perms += 1
                                            some_contribution -= 1
                                            ##########################
                                ## we want to consider that we have reached max_counts whenever there are no permutations
                                ## valid on those positions
                                if failed_perms == count_total_perms:
                                    counts = max_counts 
                                    w = window
        if C[D-1] <= c_min:
            found_max_delay_D.append(np.max(positions))
            break
        
    fcc.close()
    return C, cc                



def full_IPC_nocc_singlesys(X, st, suk,
                      max_delay, threshold,
                     max_counts=2, min_tot_deg=1, max_tot_deg=6):
    # sourcery skip: ensure-file-closed, hoist-statement-from-if, remove-redundant-if, remove-redundant-pass, remove-unnecessary-else
    """
    Calculation of the IPC.
    Args:
        X (array): collected observables for fitting.
        st (array): training input as if was fed to the reservoir (i.e., between 0 and 1).
        suk (array): test input as if was fed to the reservoir (i.e., between 0 and 1).
        max_delay (list|array):contains maximum delays to consider when calculating each degree.
                                If the threshold is hit, the maximum delay for the next degree will
                            remaining degrees will be dynamically set from the highest delay that hit the noise threshold for that degree.
        threshold (list|array): list containing the ordered lower bounds to consider a
                            contribution at a given capacity or discard it as noise.
        max_counts (int): number of times we allow to go below the threshold. Defaults to 2.
        min_tot_deg (int): minimum degree to look for contributions. Defaults to 1.
        max_tot_deg (int): maximum degree to look for contributions. Defaults to 6.
    Returns:
        C (list): capacities for each degree up to max_tot_deg.
    """

    ## Initialise files
    
    ### WE REMOVED THE FINAL'\n' FROM HERE TO PUT IT AT THE BEGINNING OF DEGREE SWITCHING
    
    total_time0 = timer()

    s_t = 2*st - 1
    s_uk = 2*suk - 1

    delayed_s_t = get_delayed_sk(s_t, max_delay[0]) ## get the delayed versions of the input, 0th max delay should be the longest
    delayed_s_uk = get_delayed_sk(s_uk, max_delay[0])

    C = np.zeros(max_tot_deg)
    cc = []

    found_max_delay_D = []  ## variable to dynamically update the maximum delay we can expect in the next degree

    for D in range(min_tot_deg, max_tot_deg+1):
        c_min = threshold[D-1]

        if found_max_delay_D != []: ##update max delay with previous degree threshold hit
            max_delay[D-1] = np.max(found_max_delay_D)
        
        max_delay_D = max_delay[D-1]

        decs = get_all_decompositions(D) ## all possible decompositions of D
        found_max_delay_D = []

        for nv in range(D):
            decs_nvar = decs[nv] ## all possible decompositions of D in nv summands
            nv = nv + 1 ##now it's the actual number of variables we have
            
            combs = list(combinations(range(max_delay_D + 1), nv))
        
            if len(combs): ### proceed only if the list of combinations is not empty, which
                ## will eventually happen if the degree D is larger than the stipulated 
                ### maximum delay

                
                for pow_list in decs_nvar: #range(len(decs_nvar)): ##pow_list IS A LIST CONTAINING THE DEGREES d_i
                    
                    combs = list(combinations(range(max_delay_D + 1), nv)) 
                    
                    ###WE MUST TREAT SEPARATELY THE CASE WITH len(pow_list) == 1 BECAUSE OF THE FUNCTION permutations
                    if len(pow_list) == 1:
                        for positions in combs:

                            ############
                            y_t = pols[pow_list][0](delayed_s_t[positions])
                            y0 = pols[pow_list][0](delayed_s_uk[positions])

                            c = get_ipc_tiny(X, y_t, y0)
                            #############print('c ', c, '\n')
                            cc.append(c)
                                
                            C[D-1] += c
                            ############
                            if c <= c_min: ## if we hit the threshold
                                cc = cc[:-1] #undo the addtion
                                found_max_delay_D.append(np.max(positions))
                                C[D-1] -= c
                                break
                    else:
                        number_distinct_minidegs = len(np.unique(pow_list))
                        w = 0 ## window in which we have failed, initially set to 0
                        counts = 0
                        some_contribution = 0

                        sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}
                                ## 0: calculate, 1: don't bother. Dictionary for optimization: avoids computing
                                # those permutations that proved to be unsufficient
                        for positions in combinations_ordered_by_window(combs):
                            window = max(positions) - min(positions)

                            #change of windows after we have exhausted the current one
                            if counts >= max_counts and window != w:
                                counts = 0
                                if some_contribution == 0: 
                                    break
                                else:
                                    some_contribution = 0
                                    sparingperms_dict = {x : 0 for x in permutations(np.arange(nv))}

    #######    WE HAVE TO INTRODUCE BY HAND THE CALCULATION OF THE FIRST ELEMENT OF A NEW WINDOW BY HAND
                                    if number_distinct_minidegs == 1:

                                        ############
                                        y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                                        y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                                        c = get_ipc_tiny(X, y_t, y0)
                                        cc.append(c)
                
                                        C[D-1] += c
                                        some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                                        ############

                                        if c <= c_min:
                                            w = window
                                            cc = cc[:-1] #undo the addtion
                                            C[D-1] -= c
                                            found_max_delay_D.append(np.max(positions))
                                            counts += 1
                                            some_contribution -= 1
                                            
                                    else:  ###we need to iterate over permutations

                                        count_total_perms = 0
                                        failed_perms = 0

                                        for rearranged_positions in admissible_orderings(pow_list, positions):
                                                                ##the resulting order already seems the most convenient
                                            perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) ) 
                                                ##it needs to be a tuple because we generated sparingperms_dict with permutations()

                                            if sparingperms_dict[perm_order_basis] == 1:
                                                count_total_perms += 1
                                                failed_perms += 1
                                                pass

                                            else:

                                                count_total_perms += 1
                                                rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                                ############
                                                y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                                y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                                c = get_ipc_tiny(X, y_t, y0)
                                                cc.append(c)

                                                C[D-1] += c
                                                some_contribution += 1
                                                ############
                                                if c <= c_min:
                                                    sparingperms_dict[perm_order_basis] = 1
                                                    w = window
                                                    cc = cc[:-1] #undo the addtion
                                                    C[D-1] -= c
                                                    found_max_delay_D.append(np.max(positions))
                                                    failed_perms += 1
                                                    some_contribution -= 1
                                                    #################print('we hit threshold, with c=', c)
                                        ## we want to consider that we have reached max_counts whenever there are no permutations
                                        ## valid on those positions
                                        if failed_perms == count_total_perms:
                                            counts = max_counts 
                                            w = window
                                            
        ####### FINAL DE PARCHE CHAPUCERO                                     


                            elif counts >= max_counts and window == w:
                                pass
                            ###go faster for the case when all minidegs are the same by just doing the first possible ordering
                            elif number_distinct_minidegs == 1:

                                ############
                                y_t = np.prod( [pols[pow_list][0](s) for s in delayed_s_t[positions] ], axis = 0 )
                                y0 = np.prod( [pols[pow_list][0](s) for s in delayed_s_uk[positions] ], axis = 0 )

                                c = get_ipc_tiny(X, y_t, y0)
                                #################print('c ', c, '\n')
                                cc.append(c)

                                C[D-1] += c
                                some_contribution += 1 ##TO SEE KNOW WHETHER WE GOT ANY CONTRIBUTION IN THIS WINDOW
                                ############
                                if c <= c_min:
                                    w = window
                                    cc = cc[:-1] #undo the addtion
                                    C[D-1] -= c
                                    found_max_delay_D.append(np.max(positions))
                                    counts += 1
                                    some_contribution -= 1
                                    #####################


                            else:  ###we need to iterate over permutations

                                count_total_perms = 0
                                failed_perms = 0

                                for rearranged_positions in admissible_orderings(pow_list, positions):
                                                        ##the resulting order already seems the most convenient
                                    perm_order_basis = tuple(get_perm_in_order_basis(rearranged_positions) ) 
                                        ##it needs to be a tuple because we generated sparingperms_dict with permutations()

                                    if sparingperms_dict[perm_order_basis] == 1:
                                        count_total_perms += 1
                                        failed_perms += 1
                                        pass
                                    else:

                                        count_total_perms += 1
                                        rearranged_positions = np.asarray(rearranged_positions, dtype = int)

                                        ############
                                        y_t = np.prod( [pols[pow_list][i](delayed_s_t[rearranged_positions][i]) for i in range(nv) ], axis = 0)
                                        y0 = np.prod( [pols[pow_list][i](delayed_s_uk[rearranged_positions][i]) for i in range(nv) ], axis = 0)

                                        c = get_ipc_tiny(X, y_t, y0)
                                        cc.append(c)
                                        

                                        C[D-1] += c
                                        some_contribution += 1
                                        ############

                                        if c <= c_min:
                                            sparingperms_dict[perm_order_basis] = 1
                                            w = window
                                            cc = cc[:-1] #undo the addtion
                                            C[D-1] -= c
                                            found_max_delay_D.append(np.max(positions))
                                            failed_perms += 1
                                            some_contribution -= 1
                                            ##########################
                                ## we want to consider that we have reached max_counts whenever there are no permutations
                                ## valid on those positions
                                if failed_perms == count_total_perms:
                                    counts = max_counts 
                                    w = window
        if C[D-1] <= c_min:
            found_max_delay_D.append(np.max(positions))
            break
        
    return C
