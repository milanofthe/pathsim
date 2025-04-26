#########################################################################################
##
##                       RANDOM NUMBER GENERATOR BLOCK (rng.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block


# NOISE SOURCE BLOCKS ===================================================================

class RNG(Block):
    """
    Generates a random output value between -1 and 1 
    from a uniform distribution.

    If no `sampling_rate` (None) is specified, every 
    simulation timestep gets a random value.

    Parameters
    ----------
    sampling_rate : float, None
        number of random samples per time unit

    Attributes
    ----------
    n_samples : int
        internal sample counter
    val : float
        internal random number state
    """

    def __init__(self, sampling_rate=None):
        super().__init__()

        self.sampling_rate = sampling_rate 
        self.n_samples = 0
        self.val = 0.0


    def __len__(self):
        return 0


    def reset(self):
        super().reset()

        #reset noise samples
        self.n_samples = 0


    def sample(self, t):
        """Sample from a normal distribution after successful timestep.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.val = 2.0*np.random.rand() - 1.0
            self.n_samples += 1


    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Note
        ----
        no direct passthrough, so the 'update' method 
        is optimized for this case        

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            deviation to previous iteration for convergence control
        """
        self.outputs[0] = self.val
        return 0.0