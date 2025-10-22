#########################################################################################
##
##                            RANDOM NUMBER GENERATOR BLOCK 
##                               (pathsim/blocks/rng.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..events.schedule import Schedule 


# BLOCKS ================================================================================

class RandomNumberGenerator(Block):
    """Generates a random output value using `numpy.random.rand`.

    If no `sampling_rate` (None) is specified, every simulation timestep gets 
    a random value. Otherwise an internal `Schedule` event is used to periodically 
    sample a random value and set the output like a sero-order-hold stage.

    Parameters
    ----------
    sampling_rate : float, None
        number of random samples per time unit

    Attributes
    ----------
    _sample : float
        internal random number state in case that 
        no `samplingrate` is provided
    Evt : Schedule
        internal event that periodically samples a random 
        value in case `samplingrate` is provided
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, sampling_rate=None):
        super().__init__()

        self.sampling_rate = sampling_rate 

        #sampling produces discrete time behavior
        if sampling_rate is None:

            #initial sample for non-discrete block
            self._sample = np.random.rand()

        else:
            
            #internal scheduled list event
            def _set(t):
                self.outputs[0] = np.random.rand()

            self.Evt = Schedule(
                t_start=0,
                t_period=sampling_rate,
                func_act=_set
                )
            self.events = [self.Evt]


    def update(self, t):
        """Setting output with random sample in case 
        of `samplingrate==None`, otherwise does nothing.

        Parameters
        ----------
        t : float
            evaluation time
        """
        if self.sampling_rate is None:
            self.outputs[0] = self._sample


    def sample(self, t, dt):
        """Generating a new random sample at each timestep 
        in case of `samplingrate==None`, otherwise does nothing.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_rate is None:
            self._sample = np.random.rand()


    def __len__(self):
        """Essentially a source-like block without passthrough"""
        return 0


class RNG(RandomNumberGenerator):

    def __init__(self, sampling_rate=None):
        super().__init__(sampling_rate)

        import warnings
        warnings.warn("'RNG' block will be deprecated with release version 1.0.0, use 'RandomNumberGenerator' instead")