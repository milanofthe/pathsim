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
    and sources by evaluating an event function (func_evt) with boolean output.

        func_evt(outputs, states, time) -> event?

    If an event is detected, some action (func_act) is performed on the states of the blocks.

        func_evt(outputs, states, time) == True -> event -> states = func_act(outputs, states, time)

    If a callback function (func_cbk) is defined, it is called with the states as args.

        func_evt(outputs, states, time) == True -> event -> func_cbk(outputs, states, time)

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        func_evt  : (callable: outputs, states, time -> bool) event function, boolean event condition
        func_act  : (callable: outputs, states, time -> states) state transform function to apply for event resolution 
        func_cbk  : (callable: outputs, states, time -> None) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self, t):
        """
        Evaluate the event function and check if condition 'g' is satisfied. 

        The event function is not differentiable, so this event always resolves 
        within the current timestep
        """

        #inactive -> quit early
        if not self._active: 
            return False, False, 1.0

        #evaluate event function
        is_event = self._evaluate(t)

        #discrete condition -> no interpolation (always resolve directly)
        return is_event, is_event, 1.0