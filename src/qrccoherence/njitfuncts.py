import numpy as np
from numpy import linalg as la
from numba import njit, float64, complex128


@njit
def fast_eigvals_eigvects(H):
    return la.eigh(H)

@njit
def fast_kron(A, B):
    return np.kron(A, B)

@njit
def fast_dot(A, B): ### not much of an improvement
    return np.dot(A, B)

@njit
def fast_ordereddot(A, B, C): ### not much of an improvement
    return np.dot(A, np.dot(B, C))

@njit
def fast_trace(A): ### not much of an improvement
    return np.trace(A)

@njit
def fast_diag(A):
    return np.diag(A)

@njit ### a bit of an improvement
def fast_least_squares(A, B):
    return la.lstsq(A, B)

@njit 
def fast_sum(A):
    return np.sum(A)

@njit 
def fast_mean(A):
    return np.mean(A)


@njit
def fast_num_combinations(n, r): ###number of combinations of n elements taken in groups of r (without repetition)
    ene = 1
    erre = 1
    enemerre = 1
    
    k = 1
    while k <= n:
        if k <= r:
            erre *= k
        if k <= n-r:
            enemerre *= k
        ene *= k
        k += 1
        
    return int(ene / (erre * enemerre) )



@njit
def get_ipc_tiny(X, y_t, y0):
    L = len(y_t)
    WW = la.lstsq(X[:L], y_t)[0].T
    y = np.sum(WW * X[L:], axis = 1)

    return 1-np.mean((y-y0)**2)/np.mean(y0**2)


#######

@njit
def C_l1(rho):
    rho_abs = np.abs(rho)
    return np.sum(rho_abs) - np.trace(rho_abs)


@njit
def VN_entropy(rho):
    eigvals, _ = la.eigh(rho)
    s=0
    for val in eigvals:
        eigenvalue = np.round(val, 6)
        if eigenvalue != 0:
            s -= val*np.log(val)
    return np.round(s, 6)




