from itertools import combinations

import numpy as np
from qrccoherence.evaluate_state import concurrence, information_content, negativity
from qrccoherence.njitfuncts import *
from qrccoherence.pauli_matrices import Ham


def initial_rho_thermal(N): ###random initial condition, no entanglement between qubits but some coherence
    r = np.random.rand(N)
    simple_rand = np.array([np.sqrt(1-ri) * np.array((1,0)) + np.sqrt(ri) * np.array((0,1)) for ri in r])
    tens = simple_rand[0]
    for i in range(N-1):
        tens = fast_kron(tens, simple_rand[i+1])
    return complex128(fast_kron(tens, tens.T).reshape(2**N, 2**N))
## initial state
def initial_state(N, initialisation='maxcoh'):
    """
    Returns initial state for the simulation.
    Args:
        N (int): number of qubits.
        initialisation (str): can be "allfacingup", "random", "maxcoh" or "maxincoh". Defaults to "maxcoh".
    """
    if initialisation == 'maxcoh':
        return 0.5**N * np.ones((2**N, 2**N))
    elif initialisation == 'maxincoh':
        rs = np.random.rand(2**N)
        rrs = rs/np.sum(rs)
        return np.diag(rrs)
    else:
        if initialisation == 'allfacingup':
            r = np.zeros(N)
        elif initialisation == 'random':
            r = np.random.rand(N)
        simple_rand = np.array([np.sqrt(1-ri) * np.array((1,0)) + np.sqrt(ri) * np.array((0,1)) for ri in r])
        tens = simple_rand[0]
        for i in range(N-1):
            tens = fast_kron(tens, simple_rand[i+1])
        return fast_kron(tens, tens.T).reshape(2**N, 2**N)
 

def monitor_information_content(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=None, dephasteps=50, 
                                initialisation='maxcoh', input_injection='mixed_z'):
    """
    Returns matrix of measured observables.
    Args:
        N (int): system size.
        J (array): 2**N x 2**N matrix, the couplings are the lower diagonal of that matrix.
        h (list|array): length N, on-site fields
        Dt (float): time between input injections
        s_wu (list|array): warm-up inputs, allows reservoir to reach a stationary state
        s_uk (list|array): input array for measurement.
        measure_wu (bool): if True, measurement is performed also during thermalisation steps. Useful to check we have reached stationary state. Defaults to False.
        decoh (function): If None, the system evolves unitarily. If given, contains the error map to apply between inputs.
                             Defaults to None.
        dephasteps (int): Number of dephasing steps. Defaults to 50.
        initialisation (str): initial state of the reservoir. Should not be a relevant variable if it has been properly thermalised. Defaults to 'maxcoh'.
        input_injection (str): can be 'mixed_z', 'mixed_x' or 'pure_zx'. Defaults to 'mixed_z'.
    Returns:
        total_mutual_information (array)
        local_coherence (array)
        total_coherence (array)
        hookup (array)
    """
    H = Ham(J, h, N)[:, :]
    eigvals, P = fast_eigvals_eigvects(H)
    rho = initial_state(N, initialisation=initialisation)
    wu_steps = len(s_wu)
    therm_T, therm_CL, therm_C, therm_M = np.zeros(wu_steps), np.zeros(wu_steps), np.zeros(wu_steps), np.zeros(wu_steps)
    if decoh is None:
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * Dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input(s, rho, evol_op, N, input_injection=input_injection)
            if measure_wu:
                therm_T[i], therm_CL[i], therm_C[i], therm_M[i] = information_content(rho, N)
        
        ## get measurement data
        L = len(s_uk)
        T, CL, C, M = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
        for i, s in enumerate(s_uk):
            rho = evolsys_input(s, rho, evol_op, N, input_injection=input_injection)
            T[i], CL[i], C[i], M[i] = information_content(rho, N)
            
            
    else:
        dt = Dt/dephasteps
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N, input_injection=input_injection)
            if measure_wu:
                therm_T[i], therm_CL[i], therm_C[i], therm_M[i] = information_content(rho, N)
        
        ## get measurement data
        L = len(s_uk)
        T, CL, C, M = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
        for i, s in enumerate(s_uk):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N, input_injection=input_injection)
            T[i], CL[i], C[i], M[i] = information_content(rho, N)
            
    return T, CL, C, M, therm_T, therm_CL, therm_C, therm_M


def monitor_concurrence(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=None, dephasteps=50, initialisation='maxcoh'):
    """
    Returns matrix of measured observables.
    Args:
        N (int): system size.
        J (array): 2**N x 2**N matrix, the couplings are the lower diagonal of that matrix.
        h (list|array): length N, on-site fields
        Dt (float): time between input injections
        s_wu (list|array): warm-up inputs, allows reservoir to reach a stationary state
        s_uk (list|array): input array for measurement.
        measure_wu (bool): if True, measurement is performed also during thermalisation steps. Useful to check we have reached stationary state. Defaults to False.
        decoh (function): If None, the system evolves unitarily. If given, contains the error map to apply between inputs.
                             Defaults to None.
        dephasteps (int): Number of dephasing steps. Defaults to 50.
        initialisation (str): initial state of the reservoir. Should not be a relevant variable if it has been properly thermalised. Defaults to 'maxcoh'.
    Returns:
        conc (array): mean concurrence of all the possible 2-qb subsystems during the stationary regime
        therm_conc (array): if measure_wu is given, returns mean concurrence of all the possible 2-qb subsystems during the thermalisation regime
    """
    H = Ham(J, h, N)[:, :]
    eigvals, P = fast_eigvals_eigvects(H)
    rho = initial_state(N, initialisation=initialisation)
    wu_steps = len(s_wu)
    therm_conc = np.zeros(wu_steps)
    therm_conc_std = np.zeros(wu_steps)
    
    L = len(s_uk)
    conc = np.zeros(L)
    conc_std = np.zeros(L)
    if decoh is None:
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * Dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input(s, rho, evol_op, N)
            if measure_wu:
                therm_conc[i], therm_conc_std[i] = concurrence(rho, N)
        
        ## get measurement data
        for i, s in enumerate(s_uk):
            rho = evolsys_input(s, rho, evol_op, N)
            conc[i], conc_std[i] = concurrence(rho, N)
    else:
        dt = Dt/dephasteps
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N)
            if measure_wu:
                therm_conc[i], therm_conc_std[i] = concurrence(rho, N)
        
        ## get measurement data
        for i, s in enumerate(s_uk):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N)
            conc[i], conc_std[i] = concurrence(rho, N)
    return conc, conc_std, therm_conc, therm_conc_std



def monitor_negativity(N, J, h, Dt, s_wu, s_uk, measure_wu=False, decoh=None, dephasteps=50, initialisation='maxcoh',
                       input_injection='mixed_z'):
    """
    Returns matrix of measured observables.
    Args:
        N (int): system size.
        J (array): 2**N x 2**N matrix, the couplings are the lower diagonal of that matrix.
        h (list|array): length N, on-site fields
        Dt (float): time between input injections
        s_wu (list|array): warm-up inputs, allows reservoir to reach a stationary state
        s_uk (list|array): input array for measurement.
        measure_wu (bool): if True, measurement is performed also during thermalisation steps. Useful to check we have reached stationary state. Defaults to False.
        decoh (function): If None, the system evolves unitarily. If given, contains the error map to apply between inputs.
                             Defaults to None.
        dephasteps (int): Number of dephasing steps. Defaults to 50.
        initialisation (str): initial state of the reservoir. Should not be a relevant variable if it has been properly thermalised. Defaults to 'maxcoh'.
        input_injection (str): can be 'mixed_z', 'mixed_x' or 'pure_zx'. Defaults to 'mixed_z'.
    Returns:
        neg (array): mean negativity of all the possible 2-qb subsystems during the stationary regime
        therm_neg (array): if measure_wu is given, returns mean negativity of all the possible 2-qb subsystems during the thermalisation regime
    """
    H = Ham(J, h, N)[:, :]
    eigvals, P = fast_eigvals_eigvects(H)
    rho = initial_state(N, initialisation=initialisation)
    wu_steps = len(s_wu)
    therm_neg = np.zeros(wu_steps)
    therm_neg_std = np.zeros(wu_steps)
    
    L = len(s_uk)
    neg = np.zeros(L)
    neg_std = np.zeros(L)
    if decoh is None:
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * Dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input(s, rho, evol_op, N, input_injection=input_injection)
            if measure_wu:
                _, _, _, _, therm_neg[i], therm_neg_std[i] = negativity(rho, N)
        
        ## get measurement data
        for i, s in enumerate(s_uk):
            rho = evolsys_input(s, rho, evol_op, N, input_injection=input_injection)
            _, _, _, _, neg[i], neg_std[i] = negativity(rho, N)
        
            
    else:
        dt = Dt/dephasteps
        evol_op = fast_dot(P, fast_dot(fast_diag( np.exp(-1j * eigvals * dt) ), P.T))
        for i, s in enumerate(s_wu):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N, input_injection=input_injection)
            if measure_wu:
                _, _, _, _, therm_neg[i], therm_neg_std[i] = negativity(rho, N)
        
        ## get measurement data
        for i, s in enumerate(s_uk):
            rho = evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N, input_injection=input_injection)
            _, _, _, _, neg[i], neg_std[i] = negativity(rho, N)
            
    return neg, neg_std, therm_neg, therm_neg_std



def function_factory(p, sZ, N): ### p is the error probability
    def Sigmamul(n_spins):  ##GET THE SIGMA MATRIX ACTING ON ARBITRARY SPINS
                                    ### n_spins must be ordered from 0 to N
        s = sZ
        indexsp = 0

        if n_spins[0] == 0:
            sigma = s
            indexsp += 1
        else:
            sigma = np.eye(2, dtype = 'complex128')

        j = 1
        while sigma.shape[0] < 2**N:
            
            if indexsp+1 < len(n_spins) and j == n_spins[indexsp]:
                sigma = fast_kron(sigma, s)
                indexsp += 1
            else:
                sigma = fast_kron(sigma, np.eye(2, dtype = 'complex128')) 
            j += 1

        return sigma
    
    order2prodsigmas = {0:np.array([np.eye(2**N, dtype = 'complex128')])}
        
    for order in range(1, N+1):

        prodsigmas = []
        for i in combinations(range(N), order):
            prodsigmas.append(Sigmamul(i))

        order2prodsigmas.update({order:prodsigmas})
        
    def decoh(rho):
        newrho = np.zeros((2**N, 2**N)) +0*1j
        for o in range(N):
            for ssgg in order2prodsigmas[o]:
                newrho += (1-p) **(N-o) * (p)**o * fast_ordereddot(ssgg, complex128(rho), ssgg)
        return newrho
    return decoh


def get_rho1_of_sk(s:float, input_injection:str):
    if input_injection == 'mixed_z':
        rho_sk = np.array([[1-s, 0], [0, s]])
    elif input_injection == 'mixed_x':
        rho_sk = np.array([[0.5, s-0.5], [s-0.5, 0.5]])
    elif input_injection == 'pure_z':
        rho_sk = np.array([[np.cos(np.pi*s/2)**2, np.cos(np.pi*s/2) * np.sin(np.pi*s/2)], 
                            [np.cos(np.pi*s/2) * np.sin(np.pi*s/2), np.sin(np.pi*s/2)**2]])  
    elif input_injection == 'pure_x':
        rho_sk = 0.5 * np.array([[1+np.sin(np.pi*s), -np.cos(np.pi*s)],
                               [-np.cos(np.pi*s), 1-np.sin(np.pi*s)]])
    return complex128(rho_sk)

def evolsys_input(s, rho, evol_op, N, input_injection='mixed_z'):  
    rho1 = get_rho1_of_sk(s, input_injection)

    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
    reduced_rho = np.einsum('ijik->jk', reshaped_rho)

    rho = fast_kron(rho1, reduced_rho)
    return fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))

def evolve_system(rho, evol_op, decoh, dephasteps:int, check_if_unitary:bool):
    if check_if_unitary: ## this is faster than checking if dephasteps==0
        # pnly step: unitary evolution
        rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))
    else:
        for _ in range(dephasteps):
            # first step: unitary evolution
            rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve

            # second step: dephasing
            rho = decoh(rho)
            rho = rho/np.trace(rho)
    return rho


def evolsys_input_with_decoherence(s, rho, evol_op, decoh, dephasteps, N, input_injection='mixed_z'):
    ##Building rho1
    rho1 = get_rho1_of_sk(s, input_injection)

    ##Taking the partial trace
    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
    reduced_rho = np.einsum('ijik->jk', reshaped_rho)

    ##Feed input
    rho = fast_kron(rho1, reduced_rho)
    ###### DEPHASING
    for _ in range(dephasteps):
        # first step: unitary evolution
        rho = fast_dot(evol_op, fast_dot(rho, np.conj(evol_op)))##Let the system evolve

        # second step: dephasing
        rho = decoh(rho)
        rho = rho/np.trace(rho)
    return rho

