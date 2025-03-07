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
    
    Notes
    -----
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

    This works by using a time dependent triangle wave for the 
    frequency and integrating it with a numerical integration 
    engine to get a continuous phase. This phase is then used 
    to evaluate a sinusoid. 

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
    """

    def __init__(self, amplitude=1, f0=1, BW=1, T=1):
        super().__init__()

        #parameters of chirp signal
        self.amplitude = amplitude
        self.f0 = f0
        self.BW = BW
        self.T = T


    def set_solver(self, Solver, **solver_args):
        
        #change solver if already initialized
        if self.engine is not None:
            self.engine = Solver.cast(self.engine, **solver_args)
            return #quit early

        #initialize the numerical integration engine with kernel
        def _f(x, u, t): return self.BW * (1 + triangle_wave(t, 1/self.T))/2
        self.engine = Solver(self.f0, _f, None, **solver_args)


    def update(self, t):
        #compute implicit balancing update
        phase = 2 * np.pi * self.engine.get()
        self.outputs[0] = self.amplitude * np.sin(phase)
        return 0.0


    def solve(self, t, dt):
        #advance solution of implicit update equation
        self.engine.solve(0.0, t, dt)

        #no error for chirp source
        return 0.0


    def step(self, t, dt):
        #compute update step with integration engine
        self.engine.step(0.0, t, dt)

        #no error control for chirp source
        return True, 0.0, 1.0