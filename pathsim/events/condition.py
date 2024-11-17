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
    This is a bidirectional zero-crossing detector. 
    
    Monitors states of solvers of stateful blocks or outputs of algebraic blocks 
    and sources by evaluating an event function (g) with boolean output.

        g(states) -> event?

    If an event is detected, some action (f) is performed on the states of the blocks.

        g(states) == True -> event -> states = f(states)

    If a callback function (h) is defined, it is called with the states as args.

        g(states) == True -> event -> h(states)

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply for event resolution 
        h         : (callable) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self):
        """
        Evaluate the event function and check if condition 'g' is satisfied. 

        The event function is not differentiable, so this event always resolves 
        within the current timestep
        """

        #inactive -> quit early
        if not self._active: 
            return False, False, 1.0

        #evaluate event function
        is_event = self._evaluate()

        #discrete condition -> no interpolation (always resolve directly)
        return is_event, is_event, 1.0