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
    
    Monitors blocks by evaluating an event function (func_evt) with boolean output.

        func_evt(blocks, time) -> event?

    If an event is detected, some action (func_act) is performed on the blocks.

        func_evt(blocks, time) == True -> event -> func_act(blocks, time)

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        func_evt  : (callable: blocks, time -> float) event function, where zeros are events
        func_act  : (callable: blocks, time -> None) action function for event resolution 
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self, t):
        """
        Evaluate the event function and check if condition 'g' is satisfied. 

        The event function is not differentiable, so this event always resolves 
        within the current timestep
        """

        #evaluate event function
        is_event = self.func_evt(self.blocks, t)

        #discrete condition -> no interpolation (always resolve directly)
        return is_event, is_event, 1.0