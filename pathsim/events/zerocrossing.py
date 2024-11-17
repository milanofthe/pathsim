#########################################################################################
##
##                                  ZERO CROSSING EVENTS
##                                (events/zerocrossing.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .event import Event


# EVENT MANAGER CLASS ===================================================================

class ZeroCrossing(Event):
    """
    Special type of event that triggers if the event function crosses zero.  
    Monitors states of solvers of stateful blocks or outputs of algebraic blocks and 
    sources by evaluating an event function (g) with scalar output and testing for 
    zero crossings (sign changes). 

    If an event is detected, some action (f) is performed on the states of the blocks.

        g(states) == 0 -> event -> states = f(states)

    If a callback function (h) is defined, it is called on the states.

        g(states) == 0 -> event -> h(states)

    Additionally the direction in which the zero-crossing occurs can be specified. 
    It basically checks the sign of the event function history from before the timestep 
    after the zero-crossing itself has been detected.

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply at events
        h         : (callable) general callaback function at event detection
        direction : (int {1, 0, -1}) direction of zero crossing for detection (0 default -> detect all)
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def __init__(self, blocks, g=None, f=None, h=None, direction=0, tolerance=1e-4):
        super().__init__(blocks, g, f, h, tolerance)

        #direction in which to cross zero for event detection
        if not direction in [1, 0, -1]: 
            raise ValueError(f"'direction' must be in [1, 0, -1]!")
        self.direction = direction


    # external methods ------------------------------------------------------------------

    def detect(self):
        """
        Evaluate the event function and check for zero-crossings
        """
        
        #evaluate event function
        result = self._evaluate()
        
        #check for zero crossing (sign change)
        is_event = np.sign(self._history) != np.sign(result)

        #definitely no event detected -> quit early
        if not is_event:
            return False, False, 1.0

        #select direction of zero crossing 
        if not (self.direction == 0 or 
            (self.direction < 0 and self._history <= 0) or 
            (self.direction > 0 and self._history >= 0)):
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(self._history) / np.clip(abs(self._history - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, ratio