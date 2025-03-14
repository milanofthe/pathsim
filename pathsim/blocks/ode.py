#########################################################################################
##
##                       ORDINARY DIFFERENTIAL EQUATION BLOCK 
##                                 (blocks/ode.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.utils import (
    dict_to_array, 
    array_to_dict
    )


# BLOCKS ================================================================================

class ODE(Block):
    """
    This block implements an ordinary differential equation (ODE) 
    defined by its right hand side

    .. math::
    
        \\dot{x}(t) = \\mathrm{func}(x(t), u(t), t)

    with inhomogenity (input) 'u' and state vector 'x'. The function 
    can be nonlinear and the ODE can be of arbitrary order. 
    The block utilizes the integration engine to solve the ODE 
    by integrating the 'func', which is the right hand side function.


    Example
    -------

    For example a linear 1st order ODE:

    .. code-block:: python
        
        from pathsim.blocks import ODE
        
        ode = ODE(lambda x, u, t: -x)


    Or something more complex like the vanderpol system, where it makes 
    sense to also specify the jacobian, which improves convergence for 
    implicit solvers but is not needed in most cases: 

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import ODE
            
        #initial condition
        x0 = np.array([2, 0])

        #van der Pol parameter
        mu = 1000

        def func(x, u, t):
            return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])

        #analytical jacobian (optional)
        def jac(x, u, t):
            return np.array(
                [[0                , 1               ], 
                 [-mu*2*x[0]*x[1]-1, mu*(1 - x[0]**2)]]
                 )
    
        #finally the block
        vdp = ODE(func, x0, jac) 
                

    Parameters
    ----------
    func : callable
        right hand side function of ODE
    initial_value : array[float]
        initial state / initial condition
    jac : callable, None
        jacobian of 'func' or 'None'
    """

    def __init__(
        self,
        func=lambda x, u, t: -x,
        initial_value=0.0,
        jac=None
        ):

        super().__init__()
          
        #right hand side function of ODE
        self.func = func

        #initial condition
        self.initial_value = initial_value

        #jacobian of 'func'
        self.jac = jac
        

    def __len__(self):
        return 0


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
            self.engine = Solver(self.initial_value, self.func, self.jac, **solver_args)
        else:
            #change solver if already initialized
            self.engine = Solver.cast(self.engine, **solver_args)
        

    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            deviation to previous iteration for convergence control
        """
        self.outputs = array_to_dict(self.engine.get())
        return 0


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
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


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
        return self.engine.step(dict_to_array(self.inputs), t, dt)