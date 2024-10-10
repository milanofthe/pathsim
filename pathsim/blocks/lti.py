#########################################################################################
##
##               LINEAR TIME INVARIANT DYNAMICAL BLOCKS (blocks/lti.py)
##
##             This module defines linear time invariant dynamical blocks
##
##                                 Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import (
    max_error_dicts,
    dict_to_array, 
    array_to_dict
    )

from ..utils.gilbert import (
    gilbert_realization
    )


# LTI BLOCKS ============================================================================

class StateSpace(Block):
    """
    This block integrates a LTI MIMO state space model with the structure

        d/dt x = A x + B u
             y = C x + D u 

    where A, B, C and D are the state space matrices, x is the state, 
    u the input and y the output vector.

    INPUTS : 
        A, B, C, D    : (numpy arrays) state space matrices
        initial_value : (array of floars) initial state / initial condition
    """

    def __init__(self, 
                 A=-1.0, B=1.0, C=-1.0, D=1.0, 
                 initial_value=None):
        super().__init__()

        #statespace matrices with input shape validation
        self.A = np.atleast_2d(A)
        self.B = np.atleast_1d(B)
        self.C = np.atleast_1d(C)
        self.D = np.atleast_1d(D)

        #get statespace dimensions
        n, _ = self.A.shape 
        if self.B.ndim == 1: n_in = 1 
        else: _, n_in = self.B.shape 
        if self.C.ndim == 1: n_out = 1 
        else: n_out, _ = self.C.shape

        #set io channels
        self.inputs  = {i:0.0 for i in range(n_in)}
        self.outputs = {i:0.0 for i in range(n_out)}

        #initial condition
        self.initial_value = np.zeros(n) if initial_value is None else initial_value


    def __len__(self):
        #check if direct passthrough exists
        return int(np.any(self.D))

    
    def set_solver(self, Solver, **solver_args):
        
        if self.engine is None:

            def _f(x, u, t): 
                return np.dot(self.A, x) + np.dot(self.B, u) 
            def _jac(x, u, t): 
                return self.A

            #initialize the integration engine with right hand side
            self.engine = Solver(self.initial_value, _f, _jac, **solver_args)

        else:

            #change solver if already initialized
            self.engine = self.engine.change(Solver, **solver_args)        


    def update(self, t):
        #compute implicit balancing update 
        prev_outputs = self.outputs.copy()
        u = dict_to_array(self.inputs)
        y = np.dot(self.C, self.engine.get()) + np.dot(self.D, u)
        self.outputs = array_to_dict(y)
        return max_error_dicts(prev_outputs, self.outputs)


    def solve(self, t, dt):
        #advance solution of implicit update equation and update outputs
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine and update outputs
        return self.engine.step(dict_to_array(self.inputs), t, dt)


class TransferFunction(StateSpace):
    """
    This block integrates a LTI (MIMO for pole residue) transfer function.

    The transfer function is defined in pole-residue form
    
        H(s) = Const + sum( Residues / (s - Poles) )

    where 'Poles' are the scalar poles of the transfer function and
    'Residues' are the possibly matrix valued (in MIMO case) residues of
    the transfer function. 'Const' has same shape as 'Residues'.

    Upon initialization, the state space realization of the transfer 
    function is computed using a minimal gilbert realization.

    The resulting statespace model of the form

        d/dt x = A x + B u
             y = C x + D u 

    is handled the same as the 'StateSpace' block, where A, B, C and D 
    are the state space matrices, x is the internal state, u the input and 
    y the output vector.

    INPUTS : 
        Poles    : (list or array of scalars) transfer function poles
        Residues : (list or array of scalars or arrays) transfer function residues
        Const    : (scalar or array) constant term of transfer function
    """

    def __init__(self, 
                 Poles=[], 
                 Residues=[], 
                 Const=0.0):

        #model parameters of transfer function in pole-residue form
        self.Const, self.Poles, self.Residues = Const, Poles, Residues

        #Statespace realization of transfer function
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        #initialize statespace model
        super().__init__(A, B, C, D)