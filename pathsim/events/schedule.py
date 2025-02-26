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
    """Subclass of base 'Event' that triggers dependent on the evaluation time. 
    
    Monitors time in every timestep and triggers periodically (period). This event
    does not have an event function as the event condition only depends on time.

        time == next_schedule_time -> event
    
    Parameters
    ----------
    t_start : float
        starting time for schedule
    t_end : float
        termination time for schedule
    t_period : float
        time period of schedule, when events are triggered
    func_act : callable
        action function for event resolution 
    tolerance : float
        tolerance to check if detection is close to actual event
    """

    def __init__(self, 
                 t_start=0, 
                 t_end=None, 
                 t_period=1, 
                 func_act=None,      
                 tolerance=1e-4):
        super().__init__(None, func_act, tolerance)
        
        #schedule times
        self.t_start = t_start
        self.t_period = t_period        
        self.t_end = t_end


    def _next(self):
        """
        return the next period break
        """
        return self.t_start + len(self._times) * self.t_period


    def buffer(self, t):
        """Buffer the current time to history
        
        Parameters
        ----------
        t : float
            buffer time
        """
        self._history = None, t


    def detect(self, t):
        """Check if the event condition is satisfied, i.e. if the 
        time period switch is within the current timestep.
        
        Parameters
        ----------
        t : float
            evaluation time for detection 
        
        Returns
        -------
        detected : bool
            was an event detected?
        close : bool
            are we close to the event?
        ratio : float
            interpolated event location ratio in timestep
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

        #are we close enough to the scheduled event?
        if abs(t_next - t) <= self.tolerance:
            return True, True, 0.0 

        #unpack history
        _, _t = self._history

        #have we already passed the event -> first timestep
        if _t >= t_next:
            return True, True, 0.0        

        #whats the timestep ratio?
        ratio = (t_next - _t) / np.clip(t - _t, 1e-18, None)

        return True, False, ratio

