#########################################################################################
##
##                           SPECIAL MIXED SIGNAL SOURCES 
##                            (blocks/mixed/sources.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .._block import Block
from ...events.schedule import Schedule


# SOURCE BLOCKS =========================================================================

class Clock(Block):

    def __init__(self, T=1, tau=0):
        super().__init__()

        self.T   = T
        self.tau = tau

        def clk_up(t):
            self.outputs[0] = 1

        def clk_down(t):
            self.outputs[0] = 0

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=clk_up
                ),
            Schedule(
                t_start=tau+T/2,
                t_period=T,
                func_act=clk_down
                )
            ]


class SquareWave(Block):

    def __init__(self, amplitude=1, frequency=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase     = phase

        def sqw_up(t):
            self.outputs[0] = self.amplitude

        def sqw_down(t):
            self.outputs[0] = -self.amplitude

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=1/frequency * phase/360,
                t_period=1/frequency,
                func_act=sqw_up
                ),
            Schedule(
                t_start=1/frequency * (phase/360 + 0.5),
                t_period=1/frequency,
                func_act=sqw_down
                )
            ]


class Step(Block):

    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.tau = tau

        def stp_up(t):
            self.outputs[0] = self.amplitude

        #internal scheduled event
        self.events = [
            Schedule(
                t_start=tau,
                t_period=tau,
                t_end=3*tau/2,
                func_act=stp_up
                )
            ]