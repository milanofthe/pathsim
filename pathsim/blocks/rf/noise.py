#########################################################################################
##
##                             SPECIAL RF NOISE SOURCES 
##                               (blocks/rf/noise.py)
##
##            this module implements some noise sources for RF simulations
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .._block import Block


# NOISE SOURCE BLOCKS ===================================================================

class WhiteNoise(Block):
    """White noise source with uniform spectral density. Samples from distribution 
    with 'sampling_rate' and holds noise values constant for time bins.

    If no 'sampling_rate' (None) is specified, every simulation timestep 
    gets a new noise values. This is the default setting.
    
    Parameters
    ----------
    spectral_density : float
        noise spectral density
    sampling_rate : float, None
        frequency with which the noise is sampled 

    Attributes
    ----------
    sigma : float
        sqrt of spectral density -> signal amplitude
    n_samples : int
        internal sample counter 
    """

    def __init__(self, spectral_density=1, sampling_rate=None):
        super().__init__()

        self.spectral_density = spectral_density
        self.sampling_rate = sampling_rate 
        self.sigma = np.sqrt(spectral_density)
        self.n_samples = 0


    def reset(self):
        #reset inputs and outputs
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0}

        #reset noise samples
        self.n_samples = 0


    def sample(self, t):
        """Sample from a normal distribution after successful timestep

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.outputs[0] = np.random.normal(0, 1)* self.sigma 
            self.n_samples += 1


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
    octaves_values : array[float]
        internal random numbers for octaves in the Voss-McCartney algorithm
    """

    def __init__(self, spectral_density=1, num_octaves=16, sampling_rate=None):
        super().__init__()

        self.spectral_density = spectral_density
        self.num_octaves = num_octaves
        self.sampling_rate = sampling_rate
        self.n_samples = 0

        # Calculate the normalization factor sigma
        self.sigma = np.sqrt(spectral_density/num_octaves)

        # Initialize the random values for each octave
        self.octave_values = np.random.normal(0, 1, self.num_octaves)


    def reset(self):
        # Reset inputs and outputs
        self.inputs  = {0: 0.0}  
        self.outputs = {0: 0.0}

        # Reset counters and octave values
        self.n_samples = 0
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
            self.outputs[0] = pink_sample * self.sigma

            


class SinusoidalPhaseNoiseSource(Block):
    """Sinusoidal source with cumulative and white phase noise

    Parameters
    ----------
    frequency : float
        frequency of the sinusoid
    amplitude : float
        amplitude of the sinusoid
    phase : float
        phase of the sinusoid
    sig_cum : float
        weight for cumulative phase noise contribution
    sig_white : float
        weight for white phase noise contribution
    samplig_rate : float
        number of samples per unit time for the internal RNG 
    """

    def __init__(self, frequency=1, amplitude=1, phase=0, sig_cum=0, sig_white=0, sampling_rate=10):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        
        self.sampling_rate = sampling_rate

        self.omega = 2 * np.pi * self.frequency

        self.sig_cum = sig_cum
        self.sig_white = sig_white

        #initial noise sampling
        self.noise_1 = np.random.normal() 
        self.noise_2 = np.random.normal() 


    def set_solver(self, Solver, **solver_args):
        
        #change solver if already initialized
        if self.engine is not None:
            self.engine = Solver.cast(self.engine, **solver_args)
            return #quit early

        #initialize the numerical integration engine with kernel
        def _f(x, u, t): return u
        self.engine = Solver(0.0, _f, None, **solver_args)


    def reset(self):
        #reset inputs and outputs
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0}

        #reset bin counter
        self.n_samples = 0
        self.t_max = 0

        #reset engine
        self.engine.reset()


    def update(self, t):

        #compute phase error
        phase_error = self.sig_white * self.noise_1 + self.sig_cum * self.engine.get()

        #set output
        self.outputs[0] = self.amplitude * np.sin(self.omega*t + self.phase + phase_error)
        return 0.0


    def sample(self, t):
        """
        Sample from a normal distribution after successful timestep.
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.noise_1 = np.random.normal() 
            self.noise_2 = np.random.normal() 
            self.n_samples += 1


    def solve(self, t, dt):
        #advance solution of implicit update equation
        self.engine.solve(self.noise_2, t, dt)
        return 0.0


    def step(self, t, dt):
        #compute update step with integration engine
        self.engine.step(self.noise_2, t, dt)

        #no error control for noise source
        return True, 0.0, 1.0