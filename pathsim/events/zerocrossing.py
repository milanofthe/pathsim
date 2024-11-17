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

from ._event import Event


# EVENT MANAGER CLASS ===================================================================

class ZeroCrossing(Event):
    """
    Subclass of base 'Event' that triggers if the event function crosses zero. 
    This is a bidirectional zero-crossing detector. 
    
    Monitors states of solvers of stateful blocks and block outputs by evaluating an 
    event function (g) with scalar output and testing for zero crossings (sign changes). 

        g(outputs, states) -> event?

    If an event is detected, some action (f) is performed on the states of the blocks.

        g(outputs, states) == 0 -> event -> states = f(states)

    If a callback function (h) is defined, it is called with the states as args.

        g(outputs, states) == 0 -> event -> h(outputs, states)

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable: outputs, states -> float) event function, where zeros are events
        f         : (callable: states -> states) state transform function to apply for event resolution 
        h         : (callable: outputs, states -> None) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self):
        """
        Evaluate the event function and check for zero-crossings
        """

        #inactive -> quit early
        if not self._active: 
            return False, False, 1.0

        #evaluate event function
        result = self._evaluate()
        
        #check for zero crossing (sign change)
        is_event = np.sign(self._history) != np.sign(result)

        #definitely no event detected -> quit early
        if not is_event:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(self._history) / np.clip(abs(self._history - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, ratio


class ZeroCrossingUp(Event):
    """
    Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from negative to positive (up). Also called
    unidirectional zero-crossing.

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply for event resolution 
        h         : (callable) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self):
        """
        Evaluate the event function and check for zero-crossings
        """
            
        #inactive -> quit early
        if not self._active: 
            return False, False, 1.0

        #evaluate event function
        result = self._evaluate()
        
        #check for zero crossing (sign change)
        is_event = np.sign(self._history) != np.sign(result)

        #no event detected or wrong direction -> quit early
        if not is_event or self._history >= 0:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(self._history) / np.clip(abs(self._history - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, ratio


class ZeroCrossingDown(Event):
    """
    Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from positive to negative (down). Also called
    unidirectional zero-crossing.

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        g         : (callable) event function, where zeros are events
        f         : (callable) state transform function to apply for event resolution 
        h         : (callable) general callaback function at event resolution
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self):
        """
        Evaluate the event function and check for zero-crossings
        """

        #inactive -> quit early
        if not self._active: 
            return False, False, 1.0
        
        #evaluate event function
        result = self._evaluate()
        
        #check for zero crossing (sign change)
        is_event = np.sign(self._history) != np.sign(result)

        #no event detected or wrong direction -> quit early
        if not is_event or self._history <= 0:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(self._history) / np.clip(abs(self._history - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, ratio