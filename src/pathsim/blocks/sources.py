#########################################################################################
##
##                            SOURCE BLOCKS (blocks/sources.py)
##
##           This module defines blocks that serve purely as inputs / sources 
##                for the simulation such as the generic 'Source' block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..events.schedule import Schedule, ScheduleList
from .._constants import TOLERANCE


# GENERIC SOURCE BLOCKS =================================================================

class Constant(Block):
    """Produces a constant output signal (SISO)
        
    Parameters
    ----------
    value : float
        constant defining block output
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, value=1):
        super().__init__()
        self.value = value


    def __len__(self):
        """No algebraic passthrough"""
        return 0
        

    def update(self, t):
        """update system equation fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence 
            control (always '0.0' because source-type)
        """
        self.outputs[0] = self.value
        return 0.0


class Source(Block):
    """Source that produces an arbitrary time dependent output, 
    defined by the func (callable).

    .. math::
    
        y(t) = \\mathrm{func}(t)


    Note
    ----
    This block is purely algebraic and its internal function (`func`) will 
    be called multiple times per timestep, each time when `Simulation._update(t)` 
    is called in the global simulation loop.


    Example
    -------
    For example a ramp:

    .. code-block:: python

        from pathsim.blocks import Source

        src = Source(lambda t : t)
    
    or a simple sinusoid with some frequency:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import Source
    
        #some parameter
        omega = 100
    
        #the function that gets evaluated
        def f(t):
            return np.sin(omega * t)

        src = Source(f)
     
    Because the `Source` block only has a single argument, it can be 
    used to decorate a function and make it a `PathSim` block. This might 
    be handy in some cases to keep definitions concise and localized 
    in the code:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import Source

        #does the same as the definition above
            
        @Source
        def src(t):
            omega = 100
            return np.sin(omega * t)

        #'src' is now a PathSim block


    Parameters
    ---------- 
    func : callable
        function defining time dependent block output
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, func=lambda t: 1):
        super().__init__()

        if not callable(func):
            raise ValueError(f"'{func}' is not callable")

        self.func = func


    def __len__(self):
        """No algebraic passthrough"""
        return 0


    def update(self, t):
        """update system equation fixed point loop 
        by evaluating the internal function 'func'

        Note
        ----
        No direct passthrough, so the `update` method 
        is optimized and has no convergence check

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs[0] = self.func(t)


# SPECIAL CONTINUOUS SOURCE BLOCKS ======================================================

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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, frequency=1, amplitude=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


    def __len__(self):
        return 0


    def _triangle_wave(self, t, f):
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


    def update(self, t):
        tau = self.phase/(2*np.pi*self.frequency)
        self.outputs[0] = self.amplitude * self._triangle_wave(t + tau, self.frequency)


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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, frequency=1, amplitude=1, phase=0):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


    def __len__(self):
        return 0


    def update(self, t):
        omega = 2*np.pi*self.frequency
        self.outputs[0] = self.amplitude * np.sin(omega*t + self.phase)


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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self, amplitude=1, f_max=1e3, tau=0.0):
        super().__init__()

        self.amplitude = amplitude
        self.f_max = f_max
        self.tau = tau


    def __len__(self):
        return 0


    def _gaussian(self, t, f_max):
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


    def update(self, t):
        self.outputs[0] = self.amplitude * self._gaussian(t-self.tau, self.f_max)


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
    sampling_rate : float
        number of samples per unit time for the internal RNG 
    
    Attributes
    ----------
    omega : float
        angular frequency of the sinusoid, derived from `frequency`
    noise_1 : float
        internal noise value sampled from normal distribution
    noise_2 : float
        internal noise value sampled from normal distribution
    n_samples : int
        bin counter for sampling
    t_max : float
        most recent sampling time, to ensure timing for sampling bins
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(
        self, 
        frequency=1, 
        amplitude=1, 
        phase=0, 
        sig_cum=0, 
        sig_white=0, 
        sampling_rate=10
        ):
        super().__init__()

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        
        self.sampling_rate = sampling_rate

        self.omega = 2 * np.pi * self.frequency

        #parameters for phase noise
        self.sig_cum = sig_cum
        self.sig_white = sig_white

        #initial noise sampling
        self.noise_1 = np.random.normal() 
        self.noise_2 = np.random.normal() 

        #bin counter
        self.n_samples = 0
        self.t_max = 0


    def __len__(self):
        return 0


    def set_solver(self, Solver, **solver_kwargs):
        #initialize the numerical integration engine 
        if self.engine is None: self.engine = Solver(0.0, **solver_kwargs)
        #change solver if already initialized
        else: self.engine = Solver.cast(self.engine, **solver_kwargs)


    def reset(self):
        super().reset()

        #reset block specific attributes
        self.n_samples = 0
        self.t_max = 0


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

        #compute phase error
        phase_error = self.sig_white * self.noise_1 + self.sig_cum * self.engine.get()

        #set output
        self.outputs[0] = self.amplitude * np.sin(self.omega*t + self.phase + phase_error)


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
        #advance solution of implicit update equation (no jacobian)
        f = self.noise_2
        self.engine.solve(f, None, dt)
        return 0.0


    def step(self, t, dt):
        #compute update step with integration engine
        f = self.noise_2
        self.engine.step(f, dt)

        #no error control for noise source
        return True, 0.0, 1.0



class ChirpPhaseNoiseSource(Block):
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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

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


    def __len__(self):
        return 0


    def _triangle_wave(self, t, f):
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


    def reset(self):
        super().reset()

        #reset 
        self.n_samples = 0
        self.t_max = 0


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


    def solve(self, t, dt):
        """advance implicit solver of implicit integration engine, evaluate 
        the triangle wave and cumulative noise RNG"""
        f = self.BW * (1 + self._triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.solve(f, None, dt)

        #no error for chirp source
        return 0.0


    def step(self, t, dt):
        """compute update step with integration engine, evaluate the triangle wave 
        and cumulative noise RNG"""
        f = self.BW * (1 + self._triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.step(f, dt)

        #no error control for chirp source
        return True, 0.0, 1.0
        

class ChirpSource(ChirpPhaseNoiseSource):

    def __init__(
        self, 
        amplitude=1, 
        f0=1, 
        BW=1, 
        T=1, 
        phase=0, 
        sig_cum=0, 
        sig_white=0, 
        sampling_rate=10):
        super().__init__(amplitude, f0, BW, T, phase, sig_cum, sig_white, sampling_rate)

        import warnings
        warnings.warn("'ChirpSource' block will be deprecated and is currently an alias, use 'ChirpPhaseNoiseSource' instead")



# SPECIAL DISCRETE SOURCE BLOCKS ========================================================

class PulseSource(Block):
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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

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


    def reset(self, t: float=None):
        """
        Resets the block state.
        
        Note
        ----
            This block has a special implementation of reset where ``t`` can be provided
            to reset the block's state to the specified time.
            This is done by changing the phase of the pulse + resetting all the internal events.

        Parameters
        ----------
            t: float, optional
                Time to reset the block state at. If None, resets to initial state.

        """
        if t:
            self._phase_start_time = t

            # event timings relative to start of cycle (tau)
            new_t_start_rise = t
            new_t_start_high = new_t_start_rise + self.t_rise
            t_plateau = self.T * self.duty
            new_t_start_fall = new_t_start_high + t_plateau
            new_t_start_low = new_t_start_fall + self.t_fall

            self.events[0].t_start = max(0.0, new_t_start_rise)
            self.events[1].t_start = max(0.0, new_t_start_high)
            self.events[2].t_start = max(0.0, new_t_start_fall)
            self.events[3].t_start = max(0.0, new_t_start_low)

            for e in self.events:
                e.reset()
        else:
            super().reset()
            self._phase = 'low'
            self._phase_start_time = self.tau

    def update(self, t):
        """Calculate the pulse output value based on the current phase.
        Performs linear interpolation during 'rising' and 'falling' phases.

        Parameters
        ----------
        t : float
            evaluation time
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


    def __len__(self):
        #no algebraic passthrough
        return 0


class Pulse(PulseSource):

    def __init__(self, amplitude=1.0, T=1.0, t_rise=0.0, t_fall=0.0, tau=0.0, duty=0.5):
        super().__init__(amplitude, T, t_rise, t_fall, tau, duty)

        import warnings
        warnings.warn("'Pulse' block will be deprecated and is currently an alias, use 'PulseSource' instead")


class ClockSource(Block):
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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

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

    def __len__(self):
        #no algebraic passthrough
        return 0


class Clock(ClockSource):

    def __init__(self, T=1, tau=0):
        super().__init__(T, tau)

        import warnings
        warnings.warn("'Clock' block will be deprecated and is currently an alias, use 'ClockSource' instead")



class SquareWaveSource(Block):
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

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

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

    def __len__(self):
        #no algebraic passthrough
        return 0


class StepSource(Block):
    """Discrete time unit step source block.
    
    Utilizes a scheduled event to set the block output 
    to the specified output levels at the defined event times.

    The arguments can be vectorial and in that case, the output is set to the 
    amplitude that corresponds to the defined delay.


    Examples
    --------

    This is how to use the source as a unit step source:

    .. code-block:: python

        from pathsim.blocks import StepSource
        
        #default, starts at 0, jumps to 1
        stp = StepSource()


    And this is how to configure it with multiple consecutive steps:

    .. code-block:: python

        from pathsim.blocks import StepSource
        
        #starts at 0, jumps to 1 at 1, jumps to -1 at 2 and jumps back to 0 at 3
        stp = StepSource(amplitude=[1, -1, 0], tau=[1, 2, 3])


    Parameters
    ----------
    amplitude : float | list[float]
        amplitude of the step signal, or amplitudes / output 
        levels of the multiple steps
    tau : float | list[float]
        delay of the step, or delays of the different steps

    Attributes
    ----------
    Evt : ScheduleList
        internal scheduled event directly accessible
    events : list[ScheduleList]
        list of interna events
    """

    #max number of ports
    _n_in_max = 0
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}
    
    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        #input type validation
        if not isinstance(amplitude, (int, float, list, np.ndarray)):
            raise ValueError(f"'amplitude' has to be float, or array of floarts, but is {type(amplitude)}")
        if not isinstance(tau, (int, float, list, np.ndarray)):
            raise ValueError(f"'tau' has to be float, or array of floarts, but is {type(tau)}!") 

        self.amplitude = amplitude if isinstance(amplitude, (list, np.ndarray)) else [amplitude]
        self.tau = tau if isinstance(tau, (list, np.ndarray)) else [tau]

        #input shape validation
        if len(self.amplitude) != len(self.tau):
            raise ValueError("'amplitude' and 'tau' must have same dimensions!")

        #internal scheduled list event
        def stp_set(t):
            idx = len(self.Evt) - 1
            self.outputs[0] = self.amplitude[idx]

        self.Evt = ScheduleList(
            times_evt=self.tau,
            func_act=stp_set
            )
        self.events = [self.Evt]

    def __len__(self):
        #no algebraic passthrough
        return 0


class Step(StepSource):

    def __init__(self, amplitude=1, tau=0.0):
        super().__init__(amplitude, tau)

        import warnings
        warnings.warn("'Step' block will be deprecated and is currently an alias, use 'StepSource' instead")