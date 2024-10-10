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
    """
    gaussian pulse with its maximum at t=0
    """
    tau = 0.5 / f_max
    return np.exp(-(t/tau)**2)


def triangle_wave(t, f):
    """
    triangle wave with amplitude '1' and frequency 'f'
    """
    return 2 * abs(t*f - np.floor(t*f + 0.5)) - 1


def square_wave(t, f):
    """
    square wave with amplitude '1' and frequency 'f'
    """
    return np.sign(np.sin(2*np.pi*f*t))


# SOURCE BLOCKS =========================================================================

class SquareWaveSource(Block):

    def __init__(self, frequency=1, amplitude=1):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency


    def update(self, t):
        self.outputs[0] = self.amplitude * square_wave(t, self.frequency)
        return 0.0


class TriangleWaveSource(Block):

    def __init__(self, frequency=1, amplitude=1):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency


    def update(self, t):
        self.outputs[0] = self.amplitude * triangle_wave(t, self.frequency)
        return 0.0


class SinusoidalSource(Block):

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

    def __init__(self, amplitude=1, f_max=1e3, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.f_max = f_max
        self.tau = tau


    def update(self, t):
        self.outputs[0] = self.amplitude * gaussian(t-self.tau, self.f_max)
        return 0.0


class StepSource(Block):

    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.tau = tau


    def update(self, t):
        self.outputs[0] = self.amplitude * float(t > self.tau)
        return 0.0


class ChirpSource(Block):

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
            self.engine = self.engine.change(Solver, **solver_args)
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
