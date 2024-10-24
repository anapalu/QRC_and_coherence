import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams["font.size"] = 24
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

FIGSIZE = (8, 5)#(10, 6)#

def retrieve_data_old(W, ipc_flag, coord, perr, system_list=np.arange(10), degree_list=np.arange(1, 6+1)): ## coord will be ideal, Sx or Sz
    path = f'./data/W{W}/{ipc_flag}/{coord}/'
    ccs = []
    Cs_tot = np.zeros(len(system_list))
    Cs = np.zeros((len(system_list), len(degree_list)))
    
    for i, nsys in enumerate(system_list):
        cs = []
        for j, deg in enumerate(degree_list):
            fname = f'cc_p{perr}_deg{deg}_nsys{nsys}.npz'
            f = np.load(path + fname)
            Cs[i, j] = f['C_deg']
            Cs_tot[i] += Cs[i, j]
            cs.append(f['cc'])
            # print(f['simulation_time'])
        ccs.append(cs)
    return Cs, ccs, Cs_tot

def retrieve_data(W, ipc_flag, coord, perr, system_list=np.arange(10), 
                  degree_list=np.arange(1, 6+1), gaussian=None, input_in_x=None): ## coord will be ideal, Sx or Sz
    path = f'./data/W{W}/{ipc_flag}/{coord}/{perr}/'
    ccs = []
    Cs_tot = np.zeros(len(system_list))
    Cs = np.zeros((len(system_list), len(degree_list)))
    
    fileend = '.npz'
    if gaussian is not None:
        fileend = f'_g{gaussian}.npz'
    if input_in_x is not None:
        fileend = fileend[:-4] + '_' + input_in_x + '.npz'
    
    for i, nsys in enumerate(system_list):
        cs = []
        for j, deg in enumerate(degree_list):
            fname = f'cc_deg{deg}_nsys{nsys}' + fileend
            if os.path.exists(path + fname):
                f = np.load(path + fname)
                Cs[i, j] = f['C_deg']
                Cs_tot[i] += Cs[i, j]
                cs.append(f['cc'])
                
            else:
                # print(path + fname, 'did not exist')
                Cs[i, j] = 0
                Cs_tot[i] += Cs[i, j]
                cs.append([0])
            # print(f['simulation_time'])
        ccs.append(cs)
    return Cs, ccs, Cs_tot



def get_path2save(W, coord, ipc_flag, add4mock=None, N=5, round=None, gaussian=None):
    """
    Returns the corresponding path + filename. If the corresponding directory didn't exist already, 
    it creates it. By default stores a .png format. coord is either x or z.
    """
    
    fileend = '.png'
    if add4mock is not None:
        fileend = f'_{add4mock}.png'
    
    if round is None and gaussian is None:
        path2save = f'./img/W{W}/{ipc_flag}/'
    elif gaussian is not None:
        path2save = f'./img_gaussian{gaussian}/W{W}/{ipc_flag}/'
    else:
        path2save = f'./img_round{round}/W{W}/{ipc_flag}/'
    
    filename2save = f'ipc_W{W}_h1_S{coord}_{ipc_flag}_N{N}_g{gaussian}'
    
    # Check whether the specified path exists or not and create it if it doesn't
    if not os.path.exists(path2save):
        os.makedirs(path2save)

    return path2save + filename2save + fileend

def retrieve_normalisation(ipc_flag):
    if ipc_flag == 'allnonloc' :#or ipc_flag == 'multiplex': ## this for V=6 and observables Z+ZZ
        normalisation = 90
    elif ipc_flag == 'nonloc_q_zz':
        normalisation = 55
    elif ipc_flag == 'quantum_nonloc' or ipc_flag == 'multiplex': ## this for V=4 and observables ZZ
        normalisation = 40
    elif ipc_flag == 'classical_locnonloc' or ipc_flag == 'nonloc_q_rand15':
        normalisation = 15
    elif ipc_flag == 'classical_nonloc' or ipc_flag == 'xx_nonloc' \
                or ipc_flag == 'zx_nonloc' or ipc_flag == 'zx_nonloc2' \
                    or ipc_flag == 'rand' or ipc_flag == 'broken_symm' \
                        or ipc_flag == 'broken_symm2' or ipc_flag == 'purex' \
                            or ipc_flag == 'purex_ZX':
        normalisation = 10
    elif ipc_flag == 'classical_loc' or ipc_flag == 'nonloc_q_rand5':
        normalisation = 5
    return normalisation
    

def join_data(W, ipc_flag, coord, get4plus=False, p_errs=[0.02, 0.15, 0.25, 0.5], 
              degree_list=np.arange(1, 6+1), gaussian=None, input_in_x= None):
    """
    Here coord is either x or z.
    """
    normalisation = retrieve_normalisation(ipc_flag)
    
    Cs, _, _ = retrieve_data(W, ipc_flag, coord='ideal', perr=0, system_list=np.arange(10), 
                             degree_list=degree_list, gaussian=gaussian, input_in_x=input_in_x)
    Csnorm = Cs/normalisation
    means = [np.mean(Csnorm, axis = 0)]
    varss = [np.std(Csnorm, axis = 0)]
       
    if get4plus:
        means[0][3] = np.sum(means[0][3:])
        means = [means[0][:4]]
        varss[0][3]  = np.sqrt(np.sum(varss[0][3:]**2))
        varss = [varss[0][:4]]
    
    
    for i, p in enumerate(p_errs[:]):
        
        Cs, _, _ = retrieve_data(W, ipc_flag, coord=f'S{coord}', perr=p, system_list=np.arange(10), \
                                                        degree_list=degree_list, gaussian=gaussian,
                                                        input_in_x=input_in_x)
        Csnorm = Cs/normalisation
        m = np.mean(Csnorm, axis = 0)
        v = np.std(Csnorm, axis = 0)
        means += [m]
        varss += [v]
        
        if get4plus:
            means[i+1][3] = np.sum(means[i+1][3:])
            means[i+1] = means[i+1][:4]
            varss[i+1][3]  = np.sqrt(np.sum(varss[i+1][3:]**2))
            varss[i+1] = varss[i+1][:4]
        
    return means, varss


def plotandsave(W, ipc_flag, coord, savefig=True):
    means, vars = join_data(W, ipc_flag, coord)
    print('mean total capacities', [np.sum(x) for x in means])
    
    
    p_error = [0, 0.02, 0.15, 0.25, 0.5]
    xaxis = p_error
    labels = [str(x) for x in xaxis] 
    width = 0.55 # the width of the bars: can also be len(x) sequence
    labs = ['1', '2', '3', '4', '5', '6+']

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(labels, 1*np.ones(len(xaxis)), 'k--', lw = 4)  

    m0 = np.array([m[0] for m in means])
    v0 = np.array([v[0] for v in vars])
    ax.bar(labels, m0, width, bottom=np.zeros(len(xaxis)), label = labs[0], yerr = v0, capsize = 4)
    bot = m0

    for i in np.arange(1, 6):
        mi = np.array([m[i] for m in means])
        vi = np.array([v[i] for v in vars])
        ax.bar(labels, mi, width, bottom=bot, label=labs[i], yerr = vi, capsize = 4)
        bot += mi

    ax.set_xlabel('$p_{err}$')
    ax.set_ylabel('IPC')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6, fancybox=True, fontsize = 20)

    if coord == 'x':   
        comment = 'bit flip'
    elif coord == 'z':
        comment = 'phase flip' 
    ax.set_title(f'S{coord} ({comment}), {ipc_flag}', pad=50)

    figname = get_path2save(W, coord, ipc_flag, add4mock=None, N=5, round=round)

    if savefig:
        fig.savefig(figname, bbox_inches='tight' )
        

def plotandsave4plus(W, ipc_flag, coord, perr_list = [0.02, 0.15, 0.25, 0.5], \
                                    savefig=True, round=None, add4mock=None, \
                                        degree_list=np.arange(1, 6+1), gaussian=None, V=4,
                                        input_in_x=None):
    
    means, vars = join_data(W, ipc_flag, coord, get4plus=True, p_errs=perr_list, \
                                    degree_list=degree_list, gaussian=gaussian,
                                    input_in_x=input_in_x)
    
    print('mean total capacities', [np.sum(x) for x in means])
    
    p_error = perr_list
    
    coord='x'
    extended_perrlist = np.append(0, perr_list)
    positions_xaxis = np.arange(len(extended_perrlist))
    ticks_xaxis = [str(x) for x in extended_perrlist] 
    
    
    xaxis = extended_perrlist
    labels = [str(x) for x in xaxis] 
    width = 0.55 # the width of the bars: can also be len(x) sequence
    labs = ['1', '2', '3', '4+']

    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # ax = axes[0]
    ax.plot(positions_xaxis, 1*np.ones(len(xaxis)), 'k--', lw = 4)  

    m0 = np.array([m[0] for m in means])
    v0 = np.array([v[0] for v in vars])
    ax.bar(positions_xaxis, m0, width, bottom=np.zeros(len(xaxis)), label = labs[0], yerr = v0, capsize = 4)
    bot = m0

    for i in np.arange(1, 4):
        mi = np.array([m[i] for m in means])
        vi = np.array([v[i] for v in vars])
        ax.bar(positions_xaxis, mi, width, bottom=bot, label=labs[i], yerr = vi, capsize = 4)
        bot += mi

    ax.set_xlabel('$p_{err}$')
    ax.set_ylabel('IPC')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6, fancybox=True, fontsize = 20)
    
    if coord == 'x':   
        comment = 'bit flip'
    elif coord == 'z':
        comment = 'phase flip' 
    if ipc_flag == 'multiplex':
        ax.set_title(f'S{coord} ({comment}), {ipc_flag} V={V}', pad=50)
    else:    
        ax.set_title(f'S{coord} ({comment}), {ipc_flag}', pad=50)
    
    ax.set_xticks(positions_xaxis)
    ax.set_xticklabels( ticks_xaxis )
    
    figname = get_path2save(W, coord, ipc_flag, add4mock=add4mock, N=5, round=round, gaussian=gaussian)
    figname = figname[:-4] + '_4plus.png'

    if savefig:
        fig.savefig(figname, bbox_inches='tight' )



def plotandsave4plus_xz(W, ipc_flag, perr_list = [0.02, 0.15, 0.25, 0.5], \
                                    savefig=True, round=None, add4mock=None, \
                                        degree_list=np.arange(1, 6+1), gaussian=None,
                                        figsize=None, title=None, input_in_x=None):
    coord='x'
    extended_perrlist = np.append(0, perr_list)
    positions_xaxis = np.arange(len(extended_perrlist))
    ticks_xaxis = [str(x) for x in extended_perrlist] 
    
    means, vars = join_data(W, ipc_flag, coord, get4plus=True, p_errs=perr_list, \
                                    degree_list=degree_list, gaussian=gaussian,
                                    input_in_x=input_in_x)
    
    print('mean total capacities Sx', [np.sum(x) for x in means])
    print('\t vars total capacities Sx', [np.round(x, 2) for x in vars])
    
    
    
    xaxis = extended_perrlist
    labels = [str(x) for x in xaxis] 
    width = 0.25 # the width of the bars: can also be len(x) sequence
    sep_bars = 0.05
    labs = ['1', '2', '3', '4+']

    if figsize is not None:
        FIGSIZE = figsize
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # ax = axes[0]
    ax.plot(positions_xaxis, 1*np.ones(len(xaxis)), 'k--', lw = 4)  

    m0 = np.array([m[0] for m in means])
    v0 = np.array([v[0] for v in vars])
    ax.bar(positions_xaxis - width/2, m0, width, bottom=np.zeros(len(xaxis)), label = labs[0], yerr = v0, capsize = 4)
    bot = m0

    for i in np.arange(1, 4):
        mi = np.array([m[i] for m in means])
        vi = np.array([v[i] for v in vars])
        ax.bar(positions_xaxis - width/2, mi, width, bottom=bot, label=labs[i], yerr = vi, capsize = 4)
        bot += mi

    ax.set_xlabel('$p_{err}$')
    ax.set_ylabel('IPC')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6, fancybox=True, fontsize = 20)
    
    # comment = 'bit flip' 
    # ax.set_title(f'S{coord} ({comment}), {ipc_flag}', pad=50)

    
    coord='z'
    
    means, vars = join_data(W, ipc_flag, coord, get4plus=True, p_errs=perr_list, \
                                    degree_list=degree_list, gaussian=gaussian,
                                    input_in_x=input_in_x)
    
    print('mean total capacities Sz', [np.sum(x) for x in means])
    print('\t vars total capacities Sz', [np.round(x, 2) for x in vars])
    
    # ax = axes[1]
    ax.plot(labels, 1*np.ones(len(xaxis)), 'k--', lw = 4)  

    m0 = np.array([m[0] for m in means])
    v0 = np.array([v[0] for v in vars])
    ax.bar(positions_xaxis + width/2 + sep_bars, m0, width, bottom=np.zeros(len(xaxis)), label = labs[0], yerr = v0, capsize = 4, color = 'C0', alpha = 0.6)
    bot = m0

    for i in np.arange(1, 4):
        mi = np.array([m[i] for m in means])
        vi = np.array([v[i] for v in vars])
        ax.bar(positions_xaxis + width/2 + sep_bars, mi, width, bottom=bot, label=labs[i], yerr = vi, capsize = 4, color = f'C{i}', alpha = 0.6)
        bot += mi

    # arr = positions_xaxis + width/2
    ax.set_xticks(positions_xaxis)
    ax.set_xticklabels( ticks_xaxis )

    if title is None:
        if gaussian is None:
            ax.set_title(f'bit flip (left bars), phase flip(right bars) \n {ipc_flag}, infinite precision', pad=50)
        else:  
            ax.set_title(f'bit flip (left bars), phase flip(right bars) \n {ipc_flag}', pad=50)
    else:
        ax.set_title(title, pad=50)
       
       
    
    ax.set_ylim(0, 1.05) 
    # ax.set_xlabel('$p_{err}$')
    # ax.set_ylabel('IPC')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=6, fancybox=True, fontsize = 20)
    fig.show()
    # comment = 'phase flip' 
    


    figname = get_path2save(W, coord, ipc_flag, add4mock=add4mock, N=5, round=round, gaussian=gaussian)
    figname = figname[:-4] + '_4plus_SxSz_singleplot.png'

    if savefig:
        print('figname', figname)
        fig.savefig(figname, bbox_inches='tight' )
