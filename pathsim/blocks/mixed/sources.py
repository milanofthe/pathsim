#########################################################################################
##
##                           SPECIAL MIXED SIGNAL SOURCES 
##                            (blocks/mixed/sources.py)
##
##                 this module implements some premade source blocks 
##                        that produce mixed signal sources
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .._block import Block
from ...events.schedule import Schedule


# HELPER FUNCTIONS ======================================================================

def square_wave(t, f):
    """
    square wave with amplitude '1' and frequency 'f'
    """
    return np.sign(np.sin(2*np.pi*f*t))


# SOURCE BLOCKS =========================================================================

class Clock(Block):

    def __init__(self, T=1, tau=0):
        super().__init__()

        self.T   = T
        self.tau = tau

        #add internal scheduled event
        self.events = [
            Schedule(
                t_start=tau,
                t_period=0.5*T
                )
        ]


    def update(self, t):
        self.outputs[0] = (1 + square_wave(t, self.frequency)) / 2
        return 0.0


class StepSource(Block):

    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.tau = tau

        #add internal scheduled event
        self.events = [
            Schedule(
                t_start=tau,
                t_period=tau,
                t_end=3*tau/2, 
                )
        ]


    def update(self, t):
        self.outputs[0] = self.amplitude * float(t > self.tau)
        return 0.0


class PulseWidthModulation(Block): pass