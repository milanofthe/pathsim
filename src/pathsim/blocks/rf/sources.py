#########################################################################################
##
##                               SPECIAL RF SOURCES 
##                             (blocks/rf/sources.py)
##
##                 this module implements some premade source blocks 
##                    that produce waveforms for RF simulations
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .._block import Block


# HELPER FUNCTIONS ======================================================================

def gaussian(t, f_max):
    """gaussian pulse with its maximum at t=0
    
    Parameters
    ----------
    t : float
        evaluation time
    f_max : float
        maximum frequency component of gaussian

    Returns
    -------
    out : float
        gaussian value
    """
    tau = 0.5 / f_max
    return np.exp(-(t/tau)**2)


def triangle_wave(t, f):
    """triangle wave with amplitude '1' and frequency 'f'

    Parameters
    ----------
    t : float
        evaluation time
    f : float
        trig wave frequency

    Returns
    -------
    out : float
        trig wave value
    """
    return 2 * abs(t*f - np.floor(t*f + 0.5)) - 1


def square_wave(t, f):
    """square wave with amplitude '1' and frequency 'f'
    
    Parameters
    ----------
    t : float
        evaluation time
    f : float
        square wave frequency

    Returns
    -------
    out : float
        square wave value
    """
    return np.sign(np.sin(2*np.pi*f*t))


# SOURCE BLOCKS =========================================================================

class SquareWaveSource(Block):
    """Source block that generates an analog square wave
    
    Note
    ----
    This block is purely analog with no internal events. 
    Not to be confused with a clock that has internal scheduled events
    
    Parameters
    ----------
    frequency : float
        frequency of the square wave
    amplitude : float
        amplitude of the square wave
    phase : float
        phase of the square wave
    """

    def __init__(self, frequency=1, amplitude=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


    def update(self, t):
        tau = self.phase/(2*np.pi*self.frequency)
        self.outputs[0] = self.amplitude * square_wave(t + tau, self.frequency)
        return 0.0


class TriangleWaveSource(Block):
    """Source block that generates an analog triangle wave
        
    Parameters
    ----------
    frequency : float
        frequency of the triangle wave
    amplitude : float
        amplitude of the triangle wave
    phase : float
        phase of the triangle wave
    """

    def __init__(self, frequency=1, amplitude=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


    def update(self, t):
        tau = self.phase/(2*np.pi*self.frequency)
        self.outputs[0] = self.amplitude * triangle_wave(t + tau, self.frequency)
        return 0.0


class SinusoidalSource(Block):
    """Source block that generates a sinusoid wave
        
    Parameters
    ----------
    frequency : float
        frequency of the sinusoid
    amplitude : float
        amplitude of the sinusoid
    phase : float
        phase of the sinusoid
    """

    def __init__(self, frequency=1, amplitude=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


    def update(self, t):
        omega = 2*np.pi*self.frequency
        self.outputs[0] = self.amplitude * np.sin(omega*t + self.phase)
        return 0.0


class GaussianPulseSource(Block):
    """Source block that generates a gaussian pulse
        
    Parameters
    ----------
    amplitude : float
        amplitude of the gaussian pulse
    f_max : float
        maximum frequency component of the gaussian pulse (steepness)
    tau : float
        time delay of the gaussian pulse 
    """

    def __init__(self, amplitude=1, f_max=1e3, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.f_max = f_max
        self.tau = tau


    def update(self, t):
        self.outputs[0] = self.amplitude * gaussian(t-self.tau, self.f_max)
        return 0.0


class StepSource(Block):
    """Source block that generates a unit step

    Parameters
    ----------
    amplitude : float
        amplitude of the step / step height
    tau : float
        time delay of the step
    """

    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.tau = tau


    def update(self, t):
        self.outputs[0] = self.amplitude * float(t > self.tau)
        return 0.0


class ChirpSource(Block):
    """Chirp source, sinusoid with frequency ramp up and ramp down.

    This works by using a time dependent triangle wave for the frequency 
    and integrating it with a numerical integration engine to get a 
    continuous phase. This phase is then used to evaluate a sinusoid.

    Additionally the chirp source can have white and cumulative phase noise. 
    Mathematically it looks like this for the contributions to the phase from 
    the triangular wave:

    .. math::

        \\varphi_t(t) = \\int_0^t \\mathrm{tri}_{f_0, B, T}(\\tau) \\, d\\tau
    
    And from the white (w) and cumulative (c) noise:

    .. math::

        \\varphi_n(t) = \\sigma_w \\, \\mathrm{RNG}_w(t) + \\sigma_c \\int_0^t \\mathrm{RNG}_c(\\tau) \\, d\\tau
    
    The phase contributions are then used to evaluate a sinusoid to get the final chirp signal:

    .. math::

        y(t) = A \\sin(\\varphi_t(t) + \\varphi_n(t) + \\varphi_0)

    Parameters
    ----------
    amplitude : float
        amplitude of the chirp signal
    f0 : float
        start frequency of the chirp signal
    BW : float
        bandwidth of the frequency ramp of the chirp signal
    T : float
        period of the frequency ramp of the chirp signal
    phase : float
        phase of sinusoid (initial)
    sig_cum : float
        weight for cumulative phase noise contribution
    sig_white : float
        weight for white phase noise contribution
    sampling_rate : float
        number of samples per unit time for the internal random number generators
    """

    def __init__(
        self, 
        amplitude=1, 
        f0=1, 
        BW=1, 
        T=1, 
        phase=0, 
        sig_cum=0, 
        sig_white=0, 
        sampling_rate=10
        ):
        super().__init__()

        #parameters of chirp signal
        self.amplitude = amplitude
        self.phase = phase
        self.f0 = f0
        self.BW = BW
        self.T = T

        #parameters for phase noise
        self.sig_cum = sig_cum
        self.sig_white = sig_white
        self.sampling_rate = sampling_rate

        #initial noise sampling
        self.noise_1 = np.random.normal() 
        self.noise_2 = np.random.normal() 

        #bin counter
        self.n_samples = 0
        self.t_max = 0


    def reset(self):
        #reset inputs and outputs
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0}

        #reset 
        self.n_samples = 0
        self.t_max = 0

        #reset engine
        self.engine.reset()


    def set_solver(self, Solver, **solver_kwargs):
        if self.engine is None:
            #initialize the numerical integration engine
            self.engine = Solver(self.f0, **solver_kwargs)
        else:
            #change solver if already initialized
            self.engine = Solver.cast(self.engine, **solver_kwargs)


    def sample(self, t):
        """Sample from a normal distribution after successful timestep 
        to update internal noise samples
        """
        if (self.sampling_rate is None or 
            self.n_samples < t * self.sampling_rate):
            self.noise_1 = np.random.normal() 
            self.noise_2 = np.random.normal() 
            self.n_samples += 1


    def update(self, t):
        """update the block output, assebble phase and evaluate the sinusoid"""
        _phase = 2 * np.pi * (self.engine.get() + self.sig_white * self.noise_1) + self.phase
        self.outputs[0] = self.amplitude * np.sin(_phase)
        return 0.0


    def solve(self, t, dt):
        """advance implicit solver of implicit integration engine, evaluate 
        the triangle wave and cumulative noise RNG"""
        f = self.BW * (1 + triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.solve(f, None, dt)

        #no error for chirp source
        return 0.0


    def step(self, t, dt):
        """compute update step with integration engine, evaluate the triangle wave 
        and cumulative noise RNG"""
        f = self.BW * (1 + triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.step(f, dt)

        #no error control for chirp source
        return True, 0.0, 1.0