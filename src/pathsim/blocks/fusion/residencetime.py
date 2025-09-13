#########################################################################################
##
##                           Blocks for residence time modeling
##                            (blocks/fusion/residencetime.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from ...ode import ODE


# BLOCKS ================================================================================

class ResidenceTime(ODE):
    """Chemical process block with residence time model.

    This block implements an internal 1st order linear ode with 
    multiple inputs, outputs and no direct passthrough.

    The internal ODE with inputs :math:`u_i` :

    .. math::
        
        \\dot{x} = - x / \\tau + \\sum_i \\beta_i u_i


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
        initial value of state
    """

    def __init__(self, tau=1, betas=None, gammas=None, initial_value=0):
    
        #time constant and input/output weights
        self.tau = tau
        self.betas = 1 if betas is None else np.array(betas)
        self.gammas = 1 if gammas is None else np.array(gammas)

        #rhs of residence time ode
        def _fn(x, u, t):
            return -x / self.tau + sum(self.betas*u)

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



class ResidenceTimeProcess(ODE):
    """Simplified version of the `ResidenceTime` model block
    with all inputs being summed equally and only the state 
    and the flux being returned to the output

    This block implements an internal 1st order linear ode with 
    multiple inputs, outputs and no direct passthrough.

    The internal ODE with inputs :math:`u_i` :

    .. math::
        
        \\dot{x} = - x / \\tau + \\sum_i u_i


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
        initial value of state
    """

    #max number of ports
    _n_out_max = 2

    #maps for input and output port labels
    _port_map_out = {"x": 0, "x/tau": 1}

    def __init__(self, tau=1, initial_value=0):
        
        #internal time constant
        self.tau = tau

        #rhs of residence time ode
        def _fn(x, u, t):
            return -x / self.tau + sum(u)

        #jacobian of rhs wrt x
        def _jc(x, u, t):
            return -1/self.tau

        #initialization just like ode block
        super().__init__(func=_fn, jac=_jc, initial_value=initial_value)


    def update(self, t):
        """update global system equation with two outputs

        Parameters
        ----------
        t : float
            evaluation time
        """
        x = self.engine.get()
        self.outputs.update_from_array([x, x/self.tau])



