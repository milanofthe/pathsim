#########################################################################################
##
##                                  ZERO CROSSING EVENTS
##                                (events/zerocrossing.py)
##
##                                    Milan Rother 2024
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
    
    Monitors blocks by evaluating an event function (func_evt) with scalar output and 
    testing for zero crossings (sign changes). 

        func_evt(blocks, time) -> event?

    If an event is detected, some action (func_act) is performed on the states of the blocks.

        func_evt(blocks, time) == 0 -> event -> func_act(blocks, time)

    INPUTS : 
        blocks    : (list[block]) list of stateful blocks to monitor
        func_evt  : (callable: blocks, time -> float) event function, where zeros are events
        func_act  : (callable: blocks, time -> None) action function for event resolution 
        tolerance : (float) tolerance to check if detection is close to actual event
    """

    def detect(self, t):
        """
        Evaluate the event function and check for zero-crossings
        """

        #evaluate event function
        result = self.func_evt(self.blocks, t)
            
        #unpack history
        _result, _t = self._history

        #no history -> no zero crossing
        if _result is None:
            return False, False, 1.0

        #check for zero crossing (sign change)
        is_event = np.sign(_result) != np.sign(result)

        #definitely no event detected -> quit early
        if not is_event:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, float(ratio)


class ZeroCrossingUp(Event):
    """
    Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from negative to positive (up). Also called
    unidirectional zero-crossing.
    """

    def detect(self, t):
        """
        Evaluate the event function and check for zero-crossings
        """
            
        #evaluate event function
        result = self.func_evt(self.blocks, t)
            
        #unpack history
        _result, _t = self._history

        #no history -> no zero crossing
        if _result is None:
            return False, False, 1.0

        #check for zero crossing (sign change)
        is_event = np.sign(_result) != np.sign(result)

        #no event detected or wrong direction -> quit early
        if not is_event or _result >= 0:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, float(ratio)


class ZeroCrossingDown(Event):
    """
    Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from positive to negative (down). Also called
    unidirectional zero-crossing.
    """

    def detect(self, t):
        """
        Evaluate the event function and check for zero-crossings
        """
        
        #evaluate event function
        result = self.func_evt(self.blocks, t)
            
        #unpack history
        _result, _t = self._history

        #no history -> no zero crossing
        if _result is None:
            return False, False, 1.0

        #check for zero crossing (sign change)
        is_event = np.sign(_result) != np.sign(result)

        #no event detected or wrong direction -> quit early
        if not is_event or _result <= 0:
            return False, False, 1.0
        
        #linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), 1e-18, None)
        
        #are we close to the actual event?
        close = abs(result) <= self.tolerance

        return True, close, float(ratio)