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

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply at events
        tolerance : (float) tolerance to check if detection is close to actual event
    
    """

    def __init__(self, blocks, g, f, tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = blocks 
    
        #event detection function
        self.g = g    

        #event action function
        self.f = f

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation history
        self._history = None

        self._times = []


    def __iter__(self):
        #just return the times at which events have been detected
        for t in self._times:
            yield t


    # internal methods ------------------------------------------------------------------

    def _get_states(self):

        #collect states from the blocks
        states = []
        for block in self.blocks:
            if block.engine is None: 
                raise ValueError(f"'{block}' has no engine!")
            states.append(block.engine.get())

        return states


    def _set_states(self, states):

        #set block states
        for state, block in zip(states, self.blocks):
            block.engine.set(state)

    
    def _evaluate(self):
        #evaluate the event function
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
        event = np.sign(result * self._history) < 0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = self._history / np.clip(self._history - result, 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) < self.tolerance

        return event, close, ratio


    def resolve(self, time=None):
        """
        Resolve the event and record the time at which it occurs. Transforms 
        the monitored states using the function 'f'.
        """

        #save the time of event resolution
        self._times.append(time)

        #transform states
        self._set_states(self.f(*self._get_states()))