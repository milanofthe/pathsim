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
    Generates a random output value beteween -1 and 1 
    from a uniform distribution.

    If no 'sampling_rate' (None) is specified, every 
    simulation timestep gets a random value.

    INPUTS : 
        sampling_rate : (float or None) number of samples per second
    """

    def __init__(self, sampling_rate=None):
        super().__init__()

        self.sampling_rate = sampling_rate 
        self.n_samples = 0


    def __len__(self):
        return 0


    def reset(self):
        #reset inputs and outputs
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0}

        #reset noise samples
        self.n_samples = 0


    def sample(self, t):
        """
        Sample from a normal distribution after successful timestep.
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.outputs[0] = 2.0*np.random.rand() - 1.0
            self.n_samples += 1
