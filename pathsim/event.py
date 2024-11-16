#########################################################################################
##
##                         EVENT MANAGER CLASS FOR EVENT DETECTION
##                                      (event.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


# EVENT MANAGER CLASS ===================================================================

class Event:
    """
    This is the heart of the event handling system based on zero crossing detection.  
    Monitors states of solvers of stateful blocks, by evaluating an event function (g) 
    with scalar output and testing for zero crossings (sign changes). If an event is 
    detected, some action (f) is performed on the states of the blocks.

        g(states) == 0 -> event -> states = f(states)

    The methods are structured such that event detection can be separated from event 
    resolution. This is required for adaptive timestep solvers to approach the event 
    and only resolve it when the event tolerance ('tolerance') is satisfied.

    If action function 'f' is not specified, the event will only be detected but other 
    than that, no transformation will be applied. For general state monitoring.

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply at events
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, blocks, g=None, f=None, tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = blocks 
    
        #event detection function
        if not callable(g): 
            raise ValueError("function 'g' needs to be callable")
        self.g = g

        #event action function (must not be callable)
        self.f = f

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation history
        self._history = None

        self._times = []


    def __iter__(self):
        """
        The '__iter__' method yields the recorded times at which events 
        are detected.
        """
        for t in self._times:
            yield t


    def __bool__(self):
        """
        The '__bool__' method is a proxy for the 'detect' method but only 
        returns wheather an the event was triggered. 
        """
        event, *_ = self.detect()
        return event



    # internal methods ------------------------------------------------------------------

    def _get_states(self):
        """
        Collect the states of the solvers (engines) of the blocks. 

        If the block doesnt have an engine, it falls back to the block 
        outputs. This enables monitoring of block outputs as well as 
        solver states.
        """
        return [b() if b.engine is None else b.engine() for b in self.blocks]


    def _set_states(self, states):
        """
        Sets the states of the solvers (engines) of the blocks. 
        
        If the block doesnt have an engine, it is skipped! Things like 
        this can be handled by the standard algebraic block interactions 
        and dont need to be handled by the event system! 
        """
        for state, block in zip(states, self.blocks):
            if block.engine is not None:
                block.engine.set(state)
            
    
    def _evaluate(self):
        """
        Evaluate the event function and return its value after casting it to float.
        """
        return float(self.g(*self._get_states()))


    # external methods ------------------------------------------------------------------

    def reset(self):
        """
        Reset the recorded event times. Resetting the history is not 
        required because of the 'buffer' method.
        """
        self._times = []


    def buffer(self):
        """
        Buffer the event function evaluation before the timestep is taken. 
        """
        self._history = self._evaluate()


    def detect(self):
        """
        Evaluate the event function and check for zero-cross
        """
        
        #evaluate event function
        result = self._evaluate()
        
        #check for zero crossing (sign change)
        is_event = np.sign(self._history) != np.sign(result)
                
        #no event detected -> quit early
        if not is_event:
            return False, False, 0.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(self._history) / np.clip(abs(self._history - result), 1e-18, None)        
        
        #are we close to the actual event?
        close = abs(result) < self.tolerance

        return True, close, ratio


    def resolve(self, time=None):
        """
        Resolve the event and record the time at which it occurs. Transforms 
        the monitored states using the function 'f' if it is defined. Otherwise 
        this just marks the location of the event.
        
        If a callback function for the event is provided, it is also called here.
        """

        #save the time of event resolution
        self._times.append(time)

        #transform states if transform available
        if self.f is not None:    
            self._set_states(self.f(*self._get_states()))