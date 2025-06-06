########################################################################################
##
##                       METHODS FOR STATESPACE REALIZATIONS
##                                (utils/gilbert.py)
##
##                                Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# STATESPACE REALIZATION ===============================================================

def gilbert_realization(Poles=[], Residues=[], Const=0.0, tolerance=1e-9): 
    """Build real valued statespace model from transfer function 
    in pole residue form by Gilberts method and an additional 
    similarity transformation to get fully real valued matrices.

    pole residue form:

    .. math::

        \\mathbf{H}(s) = \\mathmf{D} + \\sum_{n=1}^N \\frac{\\mathbf{R}_n}{s - p_n} )
    
    statespace form:
        
    .. math::

        \\mathbf{H}(s) = \\mathbf{C} (s \\mathbf{I} - \\mathbf{A})^{-1} * \\mathbf{B} + \\mathbf{H} 
    
    Notes
    -----  
    The resulting system is identical to the so-called 
    'Modal Form' and is a minimal realization.

    Parameters
    ---------- 
    Poles : array
        real and complex poles
    Residues : array
        array of real and complex residue matrices
    Const : array
        matrix for constant term
    tolerance : float
        relative tolerance for checking real poles
    
    Returns 
    -------
    A : array
        state matrix
    B : array 
        input mapping matrix
    C : array
        state to output projection matrix
    D : array, float
        direct passthrough
    """

    #make arrays
    Poles = np.atleast_1d(Poles)
    Residues = np.atleast_1d(Residues)

    #check validity of args
    if not len(Poles) or not len(Residues):
        raise ValueError("No 'Poles' and 'Residues' defined!")

    if len(Poles) != len(Residues):
        raise ValueError("Same number of 'Poles' and 'Residues' have to be given!")

    #check shape of residues for MIMO, etc
    if Residues.ndim == 1:
        N, m, n = Residues.size, 1, 1
        Residues = np.reshape(Residues, (N, m, n))
    elif Residues.ndim == 2:
        N, m, n = *Residues.shape, 1
        Residues = np.reshape(Residues, (N, m, n))
    elif Residues.ndim == 3:
        N, m, n = Residues.shape
    else:
        raise ValueError(f"shape mismatch of 'Residues': Residues.shape={Residues.shape}")

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
    D = Const * np.ones((m, n)) if np.isscalar(Const) else Const

    return  A, B, C, D
