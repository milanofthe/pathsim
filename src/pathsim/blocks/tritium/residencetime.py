#########################################################################################
##
##                           Blocks for residence time modeling
##                            (blocks/fusion/residencetime.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ..ode import ODE


# BLOCKS ================================================================================

class ResidenceTime(ODE):
    """Chemical process block with residence time model.

    This block implements an internal 1st order linear ode with 
    multiple inputs, outputs, an internal constant source term 
    and no direct passthrough.

    The internal ODE with inputs :math:`u_i` :

    .. math::
        
        \\dot{x} = - x / \\tau + \\mathrm{src} + \\sum_i \\beta_i u_i


    And the output equation for every output `i` :

    .. math::
    
        y_i = \\gamma_i x


    Parameters
    ----------
    tau : float
        residence time, inverse natural frequency (eigenvalue)
    betas: None | list[float] | np.ndarray[float]
        weights of inputs that are accumulated in state, optional
    gammas : None | list[float] | np.ndarray[float]
        weights of states (fractions) for output, optional
    initial_value : float
        initial value of state / initial quantity of process
    source_term : float
        constant source term / generation term of the process
    """

    def __init__(self, tau=1, betas=None, gammas=None, initial_value=0, source_term=0):

        #input validation
        if np.isclose(tau, 0):
            raise ValueError(f"'tau' must be nonzero but is {tau}")
    
        #time constant and input/output weights
        self.tau = tau
        self.betas = 1 if betas is None else np.array(betas)
        self.gammas = 1 if gammas is None else np.array(gammas)
        self.source_term = source_term

        #rhs of residence time ode
        def _fn(x, u, t):
            return -x/self.tau + self.source_term + sum(self.betas*u)

        #jacobian of rhs wrt x
        def _jc(x, u, t):
            return -1/self.tau

        #initialization just like ode block
        super().__init__(func=_fn, jac=_jc, initial_value=initial_value)


    def update(self, t):
        """update global system equation

        Parameters
        ----------
        t : float
            evaluation time
        """
        x = self.engine.get()
        self.outputs.update_from_array(self.gammas * x)



class Process(ResidenceTime):
    """Simplified version of the `ResidenceTime` model block
    with all inputs being summed equally and only the state 
    and the flux being returned to the output

    This block implements an internal 1st order linear ode with 
    multiple inputs, outputs and no direct passthrough.

    The internal ODE with inputs :math:`u_i` :

    .. math::
        
        \\dot{x} = - x / \\tau + \\mathrm{src} + \\sum_i u_i


    And the output equations for output `i=0` and `i=1`:

    .. math::
    
        y_0 = x

    .. math::

        y_1 = x / \\tau


    Parameters
    ----------
    tau : float
        residence time, inverse natural frequency (eigenvalue)
    initial_value : float
        initial value of state / initial quantity of process
    source_term : float
        constant source term / generation term of the process
    """

    #max number of ports
    _n_out_max = 2

    #maps for input and output port labels
    _port_map_out = {"x": 0, "x/tau": 1}

    def __init__(self, tau=1, initial_value=0, source_term=0):
        super().__init__(tau, 1, [1, 1/tau], initial_value, source_term)
