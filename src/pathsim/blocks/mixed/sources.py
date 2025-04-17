#########################################################################################
##
##                           SPECIAL MIXED SIGNAL SOURCES 
##                            (blocks/mixed/sources.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .._block import Block
from ...events.schedule import Schedule
from ..._constants import TOLERANCE


# SOURCE BLOCKS =========================================================================

class Pulse(Block):
    """Generates a periodic pulse waveform with defined rise and fall times
    using a hybrid approach with scheduled events and continuous updates.

    Scheduled events trigger phase changes (low, rising, high, falling),
    and the `update` method calculates the output value based on the
    current phase, performing linear interpolation during rise and fall.

    Parameters
    ----------
    amplitude : float, optional
        Peak amplitude of the pulse. Default is 1.0.
    T : float, optional
        Period of the pulse train. Must be positive. Default is 1.0.
    t_rise : float, optional
        Duration of the rising edge. Default is 0.0.
    t_fall : float, optional
        Duration of the falling edge. Default is 0.0.
    tau : float, optional
        Initial delay before the first pulse cycle begins. Default is 0.0.
    duty : float, optional
        Duty cycle, ratio of the pulse ON duration (plateau time only)
        to the total period T (must be between 0 and 1). Default is 0.5.
        The high plateau duration is `T * duty`.

    Attributes
    ----------
    events : list[Schedule]
        Internal scheduled events triggering phase transitions.
    _phase : str
        Current phase of the pulse ('low', 'rising', 'high', 'falling').
    _phase_start_time : float
        Simulation time when the current phase began.
    """
    def __init__(
        self, 
        amplitude=1.0, 
        T=1.0, 
        t_rise=0.0, 
        t_fall=0.0, 
        tau=0.0, 
        duty=0.5
        ):
        super().__init__()

        #input validation
        if not (T > 0):
            raise ValueError("Period T must be positive.")
        if not (0 <= t_rise):
            raise ValueError("Rise time t_rise cannot be negative.")
        if not (0 <= t_fall):
            raise ValueError("Fall time t_fall cannot be negative.")
        if not (0 <= duty <= 1):
            raise ValueError("Duty cycle must be between 0 and 1.")

        #ensure rise + high plateau + fall fits within a period
        t_plateau = T * duty
        if t_rise + t_plateau + t_fall > T:
            raise ValueError("Total pulse time (rise+plateau+fall) exceeds period T")

        #parameters
        self.amplitude = amplitude
        self.T = T
        self.t_rise = max(TOLERANCE, t_rise)
        self.t_fall = max(TOLERANCE, t_fall)
        self.tau = tau
        self.duty = duty # Duty cycle now refers to the high plateau time

        #internal state
        self._phase = 'low'
        self._phase_start_time = self.tau 

        #event timings relative to start of cycle (tau)
        t_start_rise = self.tau
        t_start_high = t_start_rise + self.t_rise 
        t_start_fall = t_start_high + t_plateau 
        t_start_low  = t_start_fall + self.t_fall 

        #define event actions (update phase and start time) 
        def _set_phase_rising(t):
            self._phase = 'rising'
            self._phase_start_time = t
            self.outputs[0] = 0.0    

        def _set_phase_high(t):
            self._phase = 'high'
            self._phase_start_time = t
            self.outputs[0] = self.amplitude

        def _set_phase_falling(t):
            self._phase = 'falling'
            self._phase_start_time = t
            self.outputs[0] = self.amplitude

        def _set_phase_low(t):
            self._phase = 'low'
            self._phase_start_time = t
            self.outputs[0] = 0.0    

        #start rising
        _E_rising = Schedule( 
            t_start=max(0.0, t_start_rise), 
            t_period=self.T, 
            func_act=_set_phase_rising
            )

        #start high plateau (end rising)
        _E_high = Schedule(
            t_start=max(0.0, t_start_high), 
            t_period=self.T, 
            func_act=_set_phase_high
            )

        #start falling
        _E_falling = Schedule( 
            t_start=max(0.0, t_start_fall), 
            t_period=self.T, 
            func_act=_set_phase_falling
            )

        #start low (end falling)
        _E_low = Schedule( 
            t_start=max(0.0, t_start_low), 
            t_period=self.T, 
            func_act=_set_phase_low
            )
        
        #scheduled events for state transitions
        self.events = [_E_rising, _E_high, _E_falling, _E_low]


    def reset(self):
        """Resets the block state."""
        super().reset()
        self._phase = 'low'
        self._phase_start_time = self.tau


    def update(self, t):
        """Calculate the pulse output value based on the current phase.
        Performs linear interpolation during 'rising' and 'falling' phases.

        Parameters
        ----------
        t : float
            current simulation time

        Returns
        -------
        error : float
            always 0.0 for this source block
        """

        #calculate output based on phase
        if self._phase == 'rising':
            _val = self.amplitude * (t - self._phase_start_time) / self.t_rise
            self.outputs[0] = np.clip(_val, 0.0, self.amplitude)
        elif self._phase == 'high':
            self.outputs[0] = self.amplitude
        elif self._phase == 'falling':
            _val = self.amplitude * (1.0 - (t - self._phase_start_time) / self.t_fall)
            self.outputs[0] = np.clip(_val, 0.0, self.amplitude)
        elif self._phase == 'low':
            self.outputs[0] = 0.0

        return 0.0 


    def __len__(self):
        #no algebraic passthrough
        return 0













class Clock(Block):
    """Discrete time clock source block.
    
    Utilizes scheduled events to periodically set 
    the block output to 0 or 1 at discrete times.

    Parameters
    ----------
    T : float
        period of the clock
    tau : float
        clock delay

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event list 
    """

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
    """Discrete time square wave source.
    
    Utilizes scheduled events to periodically set 
    the block output at discrete times.

    Parameters
    ----------
    amplitude : float
        amplitude of the square wave signal
    frequency : float
        frequency of the square wave signal
    phase : float
        phase of the square wave signal

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled events 
    """

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
    """Discrete time unit step block.
    
    Utilizes a scheduled event to set 
    the block output at the defined delay.

    Parameters
    ----------
    amplitude : float
        amplitude of the step signal
    tau : float
        delay of the step

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event 
    """

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