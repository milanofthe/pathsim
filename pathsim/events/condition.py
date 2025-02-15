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
    """
    Subclass of base 'Event' that triggers if the event function evaluates to 'True', 
    i.e. the condition is satisfied.
    
    Monitors system state by evaluating an event function (func_evt) with boolean output.

        func_evt(time) -> event?

    If an event is detected, some action (func_act) is performed on the system state.

        func_evt(time) == True -> event -> func_act(time)

    INPUTS : 
        func_evt  : (callable: time -> float) event function, where zeros are events
        func_act  : (callable: time -> None) action function for event resolution 
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self, t):
        """
        Evaluate the event function and check if condition 'g' is satisfied. 

        The event function is not differentiable, so this event always resolves 
        within the current timestep
        """

        #evaluate event function
        is_event = self.func_evt(t)

        #discrete condition -> no interpolation (always resolve directly)
        return is_event, is_event, 1.0