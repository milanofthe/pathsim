#########################################################################################
##
##                                 TIME SCHEDULED EVENTS
##                                  (events/schedule.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._event import Event


# EVENT MANAGER CLASS ===================================================================

class Schedule(Event):
    """
    Subclass of base 'Event' that triggers dependent on the evaluation time. 
    
    Monitors time in every timestep and triggers periodically (period). This event
    does not have an event function as the event condition only depends on time.

        time == schedule_time -> event

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        t_start   : (float) starting time for schedule
        t_end     : (float) termination time for schedule
        t_period  : (float) time period of schedule, when events are triggered
        func_act  : (callable: outputs, states, time -> states) state transform function to apply for event resolution 
        func_cbk  : (callable: outputs, states, time -> None) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, 
                 blocks=None, 
                 t_start=0, 
                 t_end=None, 
                 t_period=1, 
                 func_act=None, 
                 func_cbk=None,         
                 tolerance=1e-4):
        
        #blocks to monitor for events
        self.blocks = [] if blocks is None else blocks

        #schedule times
        self.t_start = t_start
        self.t_end = t_end
        self.t_period = t_period

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


    def _evaluate(self, t):
        return None


    def _next(self):
        """
        return the next period break
        """
        return self.t_start + len(self._times) * self.t_period


    def detect(self, t):
        """
        Check if the event condition is satisfied, i.e. if the time period 
        switch is within the current timestep.

        INPUTS :
            t : (float) evaluation time for detection 
        
        RETURNS : 
            detected : (bool) was an event detected?
            close    : (bool) are we close to the event?
            ratio    : (float) interpolated event location ratio in timestep
        """

        #get next period break
        t_next = self._next()

        #end time reached? -> deactivate event, quit early
        if self.t_end is not None and t_next > self.t_end:
            self.off()
            return False, False, 1.0
        
        #no event -> quit early
        if t_next > t:
            return False, False, 1.0

        #unpack history
        _, _t = self._history

        #are we close enough to the scheduled event?
        close = abs(t_next - t) < self.tolerance

        #whats the timestep ratio?
        ratio = (t_next - _t) / np.clip(t - _t, 1e-18, None)

        return True, close, ratio
        

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