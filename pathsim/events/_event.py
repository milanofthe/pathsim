#########################################################################################
##
##                         EVENT MANAGER CLASS FOR EVENT DETECTION
##                                   (events/_event.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


# EVENT MANAGER CLASS ===================================================================

class Event:
    """
    This is the base class of the event handling system.

    Monitors states of solvers of stateful blocks or outputs of algebraic blocks and 
    sources by evaluating an event function (g) with scalar output.

        g(states) -> event?
    
    If an event is detected, some action (f) is performed on the states of the blocks.

        event -> states = f(states)

    If a callback function (h) is defined, it is called with the states as args.

        event -> h(states)

    The methods are structured such that event detection can be separated from event 
    resolution. This is required for adaptive timestep solvers to approach the event 
    and only resolve it when the event tolerance ('tolerance') is satisfied.

    If no action functions 'f', 'h' are specified, the event will only be detected but 
    other than that, no transformation will be applied. For general state monitoring.

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, gets evaluated for event detection
        f         : (callable) state transform function to apply for event resolution 
        h         : (callable) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, blocks=None, g=None, f=None, h=None, tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = [] if blocks is None else blocks
    
        #event detection function
        if not callable(g): 
            raise ValueError("function 'g' needs to be callable")
        self.g = g

        #event action function -> state transform (must not be callable)
        self.f = f

        #event action function -> general callback (must not be callable)
        self.h = h

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation history
        self._history = None

        #recording the event times
        self._times = []

        #flag for active event checking
        self._active = True


    def __str__(self):
        return self.__class__.__name__


    def __len__(self):
        """
        Return the number of detected (or rather resolved) events.
        """
        return len(self._times)


    def __iter__(self):
        """
        Yields the recorded times at which events are detected.
        """
        for t in self._times:
            yield t


    def __bool__(self):
        """
        Proxy for the 'detect' method but only returns if the event was triggered. 
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

    def on(self): self._active = True
    def off(self): self._active = False


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
        Evaluate the event function and decide if an event has occured. 
        Can also use the history of the event function evaluation from 
        before the timestep.

        NOTE : 
            This does nothing and needs to be implemented for specific events!!!
        
        RETURNS : 
            detected : (bool) was an event detected?
            close    : (bool) are we close to the event?
            ratio    : (float) interpolated event location ratio in timestep
        """

        return False, False, 1.0
        

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

        #general callback function
        if self.h is not None:
            self.h(*self._get_states())