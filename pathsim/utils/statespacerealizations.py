########################################################################################
##
##                       METHODS FOR STATESPACE REALIZATIONS
##                        (utils/statespacerealizations.py)
##
##                                Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# STATESPACE REALIZATION ===============================================================

def gilbert_realization(Poles, Residues, Const=0.0, tolerance=1e-9):
        
    """
    Build real valued statespace model from transfer function 
    in pole residue form by Gilberts method and an additional 
    similarity transformation to get fully real valued matrices.

    pole residue form:
        H(s) = Const + sum( Residues / (s - Poles) )
    
    statespace form:
        H(s) = C * (s*I - A)^-1 * B + D 
    
    NOTE :  
        The resulting system is identical to the so-called 
        'Modal Form' and is a minimal realization.

    INPUTS : 
        Poles     : (array) real and complex poles
        Residues  : (array) array of real and complex residue matrices
        Const     : (array) matrix for constant term
        tolerance : (float) relative tolerance for checking real poles
    """

    #dimension checks for residue matrices
    if isinstance(Residues, (list, tuple)) or Residues.size == len(Residues):
        Residues = np.reshape(Residues, (len(Residues), 1, 1))

    #get dimensions for MIMO
    N, m, n = Residues.shape

    #initialize companion matrix
    a = np.zeros((N, N))
    b = np.zeros(N)
        
    #residues
    C = np.ones((m, n*N))
    
    #go through poles and handle conjugate pairs
    _Poles, _Residues = [], []
    for p, R in zip(Poles, Residues):

        #real pole
        if np.isreal(p) or abs(np.imag(p) / np.real(p)) < tolerance:
            _Poles.append(p.real)
            _Residues.append(R.real)

        #complex conjugate pair
        elif np.imag(p) > 0.0:
            _Poles.extend([p, np.conj(p)])
            _Residues.extend([R, np.conj(R)])

    #build real companion matrix from the poles
    p_old = 0.0
    for k, (p, R) in enumerate(zip(_Poles, _Residues)):
        
        #check if complex conjugate
        is_cc = (p.imag != 0.0 and p == np.conj(p_old))
        p_old = p
        
        a[k,k] = np.real(p)
        b[k] = 1.0
        if is_cc:
            a[k, k-1] = - np.imag(p)
            a[k-1, k] = np.imag(p)
            b[k]   = 0.0
            b[k-1] = 2.0
            
        #iterate columns of residue
        for i in range(n):
            C[:,k+N*i] = np.imag(R[:,i]) if is_cc else np.real(R[:,i])  
                
    #build block diagonal
    A = np.kron(np.eye(n, dtype=float), a)
    B = np.kron(np.eye(n, dtype=float), b).T
    D = Const

    return  A, B, C, D
