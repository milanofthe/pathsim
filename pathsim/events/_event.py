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
    
    Monitors blocks by evaluating an event function (func_evt) with scalar output.

        func_evt(blocks, time) -> event?

    If an event is detected, some action (func_act) is performed on the states of the blocks.

        func_evt(blocks, time) == True -> event -> func_act(blocks, time)

    The methods are structured such that event detection can be separated from event 
    resolution. This is required for adaptive timestep solvers to approach the event 
    and only resolve it when the event tolerance ('tolerance') is satisfied.

    If no action function (func_act) is specified, the event will only be detected but other 
    than that, no action will be triggered. For general state monitoring.    

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        func_evt  : (callable: blocks, time -> float) event function, where zeros are events
        func_act  : (callable: blocks, time -> None) action function for event resolution 
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, 
                 blocks=None, 
                 func_evt=None, 
                 func_act=None, 
                 tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = [] if blocks is None else blocks
    
        #event detection function
        self.func_evt = func_evt

        #event action function -> event resolution (must not be callable)
        self.func_act = func_act

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation and evaluation time history (eval, time)
        self._history = None, 0.0

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
        return self._active


    # external methods ------------------------------------------------------------------

    def on(self): self._active = True
    def off(self): self._active = False


    def reset(self):
        """
        Reset the recorded event times. Resetting the history is not 
        required because of the 'buffer' method.
        """
        self._history = None, 0.0
        self._times = []


    def buffer(self, t):
        """
        Buffer the event function evaluation before the timestep is taken and the evaluation time. 
        
        INPUTS :
            t : (float) evaluation time for buffering history
        """
        if self.func_evt is not None:
            self._history = self.func_evt(self.blocks, t), t


    def detect(self, t):
        """
        Evaluate the event function and decide if an event has occured. 
        Can also use the history of the event function evaluation from 
        before the timestep.

        INPUTS :
            t : (float) evaluation time for detection 

        NOTE : 
            This does nothing and needs to be implemented for specific events!!!
        
        RETURNS : 
            detected : (bool) was an event detected?
            close    : (bool) are we close to the event?
            ratio    : (float) interpolated event location ratio in timestep
        """

        return False, False, 1.0
        

    def resolve(self, t):
        """
        Resolve the event and record the time (t) at which it occurs. 
        Resolves event using the action function (func_act) if it is defined. 

        Otherwise this just marks the location of the event in time.

        INPUTS :
            t : (float) evaluation time for event resolution
        """

        #save the time of event resolution
        self._times.append(t)

        #action function for event resolution
        if self.func_act is not None:
            self.func_act(self.blocks, t)