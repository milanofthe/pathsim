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

from ..utils.register import Register
from ..utils.gilbert import gilbert_realization

from ..optim.operator import DynamicOperator


# LTI BLOCKS ============================================================================

class StateSpace(Block):
    """
    This block defines a linear time invariant (LTI) multi input multi output (MIMO) 
    state space model with the structure

    .. math::

        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    where `A`, `B`, `C` and `D` are the state space matrices, `x` is the state, 
    `u` the input and `y` the output vector.
    
    Example
    -------
    A SISO state space block with two internal states can be initialized 
    like this:

    .. code-block:: python

        S = StateSpace(
            A=-np.eye(2), 
            B=np.ones((2, 1)), 
            C=np.ones((1, 2)), 
            D=1.0
            )

    and a MIMO (2 in, 2 out) state space block with three internal states 
    can be initialized like this:

    .. code-block:: python

        S = StateSpace(
            A=-np.eye(3), 
            B=np.ones((3, 2)), 
            C=np.ones((2, 3)), 
            D=np.ones((2, 2))
            )

    Parameters
    ----------
    A, B, C, D : array_like
        state space matrices
    initial_value : array_like, None
        initial state / initial condition

    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for state equation
    op_alg : DynamicOperator
        internal algebraic operator for mapping to outputs
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
        self.inputs = Register(n_in)
        self.outputs = Register(n_out)

        #initial condition and shape validation
        if initial_value is None:
            self.initial_value = np.zeros(n)
        else: 
            self.initial_value = np.atleast_1d(initial_value)

        #operators
        self.op_dyn = DynamicOperator(
            func=lambda x, u, t: np.dot(self.A, x) + np.dot(self.B, u),
            jac_x=lambda x, u, t: self.A,
            jac_u=lambda x, u, t: self.B
            )
        self.op_alg = DynamicOperator(
            func=lambda x, u, t: np.dot(self.C, x) + np.dot(self.D, u),
            jac_x=lambda x, u, t: self.C,
            jac_u=lambda x, u, t: self.D
            )


    def __len__(self):
        #check if direct passthrough exists
        return int(np.any(self.D)) if self._active else 0


    def set_solver(self, Solver, **solver_args):
        """set the internal numerical integrator

        Parameters
        ----------
        Solver : Solver
            numerical integration solver class
        solver_args : dict
            parameters for solver initialization
        """
        
        if self.engine is None:
            #initialize the integration engine with right hand side
            self.engine = Solver(self.initial_value, **solver_args)

        else:
            #change solver if already initialized
            self.engine = Solver.cast(self.engine, **solver_args)


    def solve(self, t, dt):
        """advance solution of implicit update equation of the solver

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        x, u = self.engine.get(), self.inputs.to_array()
        f, J = self.op_dyn(x, u, t), self.op_dyn.jac_x(x, u, t)
        return self.engine.solve(f, J, dt)


    def step(self, t, dt):
        """compute timestep update with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        x, u = self.engine.get(), self.inputs.to_array()
        f = self.op_dyn(x, u, t)
        return self.engine.step(f, dt)


class TransferFunction(StateSpace):
    """This block defines a LTI (MIMO for pole residue) transfer function.

    The transfer function is defined in pole-residue form

    .. math::
        
        \\mathbf{H}(s) = \\mathbf{C} + \\sum_n^N \\frac{\\mathbf{R}_n}{s - p_n}

    where 'Poles' are the scalar (possibly complex conjugate) poles of the 
    transfer function and 'Residues' are the possibly matrix valued (in MIMO case) 
    and complex conjugate residues of the transfer function. 'Const' has same 
    shape as 'Residues'.

    Upon initialization, the state space realization of the transfer 
    function is computed using a minimal gilbert realization.

    The resulting state space model of the form

    .. math::
        
        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
    are the state space matrices, `x` is the internal state, `u` the input and 
    `y` the output vector.
        
    Parameters
    ----------
    Poles : array
        transfer function poles
    Residues : array
        transfer function residues
    Const : array, float
        constant term of transfer function
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