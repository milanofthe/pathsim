#########################################################################################
##
##                             SPECIAL RF NOISE SOURCES 
##                                (blocks/rf/noise.py)
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
    """
    White noise source with uniform spectral density. Samples from distribution 
    with 'sampling_rate' and holds noise values constant for time bins.

    If no 'sampling_rate' (None) is specified, every simulation timestep 
    gets a new noise values. This is the default setting.

    INPUTS : 
        spectral_density : (float) noise spectral density
        sampling_rate    : (float or None) frequency with which the noise is sampled 
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
        """
        Sample from a normal distribution after successful timestep.
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.outputs[0] = np.random.normal(scale=self.sigma) 
            self.n_samples += 1


class OneOverFNoise(Block):
    """
    1/f noise source that is realized by integrating white noise using a 
    numerical integrator. Samples from distribution with 'sampling_rate' 
    and holds noise values constant for time bins.

    If no 'sampling_rate' (None) is specified, every simulation timestep 
    gets a new noise values. This is the default setting.

    INPUTS : 
        spectral_density : (float) noise spectral density
        sampling_rate     : (float or None) frequency with which the noise is sampled 
    """

    def __init__(self, spectral_density=1, sampling_rate=None):
        super().__init__()

        #parameters of noise signal
        self.spectral_density = spectral_density
        self.sampling_rate = sampling_rate 
        self.sigma = np.sqrt(spectral_density)
        self.white_noise_value = 0.0
        self.n_samples = 0


    def set_solver(self, Solver, **solver_args):
        
        #change solver if already initialized
        if self.engine is not None:
            self.engine = self.engine.change(Solver, **solver_args)
            return #quit early

        #initialize the numerical integration engine with kernel
        def _f(x, u, t): return u
        self.engine = Solver(0.0, _f, None, **solver_args)


    def reset(self):
        #reset inputs and outputs
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0}

        #reset noise samples
        self.white_noise_value = 0.0
        self.n_samples = 0

        #reset engine
        self.engine.reset()


    def update(self, t):
        #set outputs
        self.outputs[0] = self.engine.get()
        return 0.0


    def sample(self, t):
        """
        Sample from a normal distribution after successful timestep.
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.white_noise_value = np.random.normal(scale=self.sigma) 
            self.n_samples += 1


    def solve(self, t, dt):
        #advance solution of implicit update equation
        self.engine.solve(self.white_noise_value, t, dt)
        return 0.0


    def step(self, t, dt):
        #compute update step with integration engine
        self.engine.step(self.white_noise_value, t, dt)

        #no error control for noise source
        return True, 0.0, 0.0, 1.0


class SinusoidalPhaseNoiseSource(Block):

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
            self.engine = self.engine.change(Solver, **solver_args)
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
        return True, 0.0, 0.0, 1.0