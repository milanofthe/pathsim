#########################################################################################
##
##                             TIME DOMAIN NOISE SOURCES 
##                                (blocks/noise.py)
##
##                              Milan Rother 2024/25
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block


# NOISE SOURCE BLOCKS ===================================================================

class WhiteNoise(Block):
    """White noise source with uniform spectral density. Samples from distribution 
    with 'sampling_rate' and holds noise values constant for time bins.

    If no 'sampling_rate' (None) is specified, every simulation timestep 
    gets a new noise value. This is the default setting.
    
    Parameters
    ----------
    spectral_density : float
        noise spectral density
    noise : float
        internal noise value
    sampling_rate : float, None
        frequency with which the noise is sampled 

    Attributes
    ----------
    sigma : float
        sqrt of spectral density -> signal amplitude
    n_samples : int
        internal sample counter 
    """
    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, spectral_density=1, sampling_rate=None):
        super().__init__()

        #clear all inputs because its a source block
        self.inputs.clear()

        self.spectral_density = spectral_density
        self.sampling_rate = sampling_rate 
        self.sigma = np.sqrt(spectral_density)
        self.n_samples = 0
        self.noise = 0.0


    def __len__(self):
        return 0


    def reset(self):
        super().reset()

        #reset noise samples
        self.n_samples = 0
        self.noise = 0.0


    def sample(self, t):
        """Sample from a normal distribution after successful timestep

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.noise = np.random.normal(0, 1) * self.sigma 
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
        """
        self.outputs[0] = self.noise


class PinkNoise(Block):
    """Pink noise (1/f) source using the Voss-McCartney algorithm.

    Samples from distribution with 'sampling_rate' and generates noise
    with a power spectral density inversely proportional to frequency.
    
    Parameters
    ----------
    spectral_density : float
        Desired noise spectral density
    num_octaves : int
        Number of octaves (levels of randomness)
    sampling_rate : float, None
        Frequency with which the noise is sampled 

    Attributes
    ----------
    sigma : float
        sqrt of spectral density normalized to number of octaves
    n_samples : int
        internal sample counter 
    noise : float
        internal noise value
    octaves_values : array[float]
        internal random numbers for octaves in the Voss-McCartney algorithm
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, spectral_density=1, num_octaves=16, sampling_rate=None):
        super().__init__()
        
        #clear all inputs because its a source block
        self.inputs.clear()

        self.spectral_density = spectral_density
        self.num_octaves = num_octaves
        self.sampling_rate = sampling_rate
        self.n_samples = 0
        self.noise = 0.0

        # Calculate the normalization factor sigma
        self.sigma = np.sqrt(spectral_density/num_octaves)

        # Initialize the random values for each octave
        self.octave_values = np.random.normal(0, 1, self.num_octaves)


    def __len__(self):
        return 0


    def reset(self):
        super().reset()

        #reset counters and octave values
        self.n_samples = 0
        self.noise = 0.0
        self.octave_values = np.random.normal(0, 1, self.num_octaves)


    def sample(self, t):
        """Generate a new pink noise sample at 't' using 
        the Voss-McCartney algorithm.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):

            # Increment the counter
            self.n_samples += 1

            # Use bitwise operations to determine which octaves to update
            mask, idx = self.n_samples, 0
            while mask & 1 == 0 and idx < self.num_octaves:
                mask >>= 1
                idx += 1

            # Update the selected octave with a new random value
            if idx < self.num_octaves:    
                self.octave_values[idx] = np.random.normal(0, 1)

            # Sum the octave values to produce the pink noise sample
            pink_sample = np.sum(self.octave_values)

            # Normalize by sigma to maintain consistent amplitude
            self.noise = pink_sample * self.sigma

            
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
        """
        self.outputs[0] = self.noise