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
    def __init__(self, tau=1, betas=None, gammas=None, initial_value=0):
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
        self.tau = tau
        self.betas = 1 if betas is None else np.array(betas)
        self.gammas = 1 if gammas is None else np.array(gammas)

        #rhs of residence time ode
        def _fn(x, u, t):
        	return -x / self.tau + sum(self.betas*u)

        #initialization just like ode block
        super().__init__(func=_fn, initial_value=initial_value)


    def update(self, t):
    	"""update global system equation

        Parameters
        ----------
        t : float
            evaluation time
        """
        x = self.engine.get()
        self.outputs.update_from_array(self.gammas * x)

