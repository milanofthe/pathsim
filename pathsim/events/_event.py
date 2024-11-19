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
    sources by evaluating an event function (func_evt) with scalar output.

        func_evt(outputs, states, time) -> event?

    If an event is detected, some action (func_act) is performed on the states of the blocks.

        func_evt(outputs, states, time) == True -> event -> states = func_act(outputs, states, time)

    If a callback function (func_cbk) is defined, it is called with the states as args.

        func_evt(outputs, states, time) == True -> event -> func_cbk(outputs, states, time)

    The methods are structured such that event detection can be separated from event 
    resolution. This is required for adaptive timestep solvers to approach the event 
    and only resolve it when the event tolerance ('tolerance') is satisfied.

    If no action function (func_act), or callback (func_cbk) are specified, the event will only be 
    detected but other than that, no transformation will be applied. For general state monitoring.    

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        func_evt  : (callable: outputs, states, time -> float) event function, where zeros are events
        func_act  : (callable: outputs, states, time -> states) state transform function to apply for event resolution 
        func_cbk  : (callable: outputs, states, time -> None) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, 
                 blocks=None, 
                 func_evt=None, 
                 func_act=None, 
                 func_cbk=None, 
                 tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = [] if blocks is None else blocks
    
        #event detection function
        if not callable(func_evt): 
            raise ValueError("function 'func_evt' needs to be callable")
        self.func_evt = func_evt

        #event action function -> state transform (must not be callable)
        self.func_act = func_act

        #event action function -> general callback (must not be callable)
        self.func_cbk = func_cbk

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation and evaluation time history (eval, time)
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


    # internal methods ------------------------------------------------------------------

    def _get(self):
        """
        Collect the outputs of the blocks and the states of the solvers 
        (engines) of the blocks as tuples.

            (outputs, states), ...

        This enables monitoring of block outputs as well as solver states.

        RETURNS : 
            outputs : (list[array, float]) outputs of the blocks
            states  : (list[array, float]) internal states of the blocks
        """

        #no blocks to watch -> no states and outputs
        if not self.blocks: return [], []
        return zip(*[block() for block in self.blocks])


    def _set_states(self, states):
        """
        Sets the states of the solvers (engines) of the blocks. 
        
        If the block doesnt have an engine, it is skipped! Things like 
        this can be handled by the standard algebraic block interactions 
        and dont need to be handled by the event system! 

        INPUTS :
            states : (list[array, float]) new internal states of the blocks 
        """
        for state, block in zip(states, self.blocks):
            if block.engine: block.engine.set(state)
            
    
    def _evaluate(self, t):
        """
        Evaluate the event function and return its value.

        INPUTS :
            t : (float) evaluation time for event function
        """
        outputs, states = self._get()
        return self.func_evt(outputs, states, t)


    # external methods ------------------------------------------------------------------

    def on(self): self._active = True
    def off(self): self._active = False


    def reset(self):
        """
        Reset the recorded event times. Resetting the history is not 
        required because of the 'buffer' method.
        """
        self._times = []


    def buffer(self, t):
        """
        Buffer the event function evaluation before the timestep is taken and the evaluation time. 
        
        INPUTS :
            t : (float) evaluation time for buffering history
        """
        self._history = (self._evaluate(t), t)


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
        Resolve the event and record the time (t) at which it occurs. Transforms 
        the monitored states using the action function (func_act) if it is defined. 
        Otherwise this just marks the location of the event in time.
        
        If a callback function (func_cbk) for the event is provided, it is also called here.

        INPUTS :
            t : (float) evaluation time for event resolution
        """

        #save the time of event resolution
        self._times.append(t)

        #transform states if transform available
        if self.func_act is not None:
            outputs, states = self._get()
            self._set_states(self.func_act(outputs, states, t))

        #general callback function 
        if self.func_cbk is not None:
            outputs, states = self._get()
            self.func_cbk(outputs, states, t)

