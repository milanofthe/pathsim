#########################################################################################
##
##                                    CONDITION EVENTS
##                                 (events/condition.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._event import Event


# EVENT MANAGER CLASS ===================================================================

class Condition(Event):
    """Subclass of base 'Event' that triggers if the event function evaluates to 'True', 
    i.e. the condition is satisfied.
    
    Monitors system state by evaluating an event function (func_evt) with boolean output.

        func_evt(time) -> event?

    If an event is detected, some action (func_act) is performed on the system state.

        func_evt(time) == True -> event -> func_act(time)
    """

    def detect(self, t):
        """
        Evaluate the event function and check if condition 'g' is satisfied. 

        The event function is not differentiable, so this event always resolves 
        within the current timestep

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
            interpolated event location as ratio of timestep
        """

        #evaluate event function
        is_event = self.func_evt(t)

        #discrete condition -> no interpolation (always resolve directly)
        return is_event, is_event, 1.0