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
    The event is considered detected when the event function evaluates to 'True' for the 
    first time. Subsequent evaluations to 'True' are not considered unless the event is reset.
    
    .. code-block::

        func_evt(time) -> event?

    If an event is detected, some action (func_act) is performed on the system state.

    .. code-block::

        func_evt(time) == True -> event -> func_act(time)

    Note
    ----
    Condition event functions evaluate to boolean and are therefore not smooth. 
    Therefore uses bisection method for event location instead of secant method.

    Example
    -------
    Initialize a conditional event handler like this:

    .. code-block:: python

        #define the event function
        def evt(t):
            return t > 10
        
        #define the action function (callback)
        def act(t):
            #do something at event resolution
            pass
    
        #initialize the event manager
        E = Condition(
            func_evt=evt,  #the event function
            func_act=act   #the action function
            )    

    """

    def detect(self, t):
        """
        Evaluate the event function and check if condition is satisfied. 

        The event function is not differentiable, so we use bisection to 
        narrow down its location to some tolerance.

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
            adjust timestep to locate event
        """

        #unpack history
        _result, _t = self._history

        #evaluate event function
        result = self.func_evt(t)

        #check if interval narrowed down sufficiently
        close = result and (t - _t) < self.tolerance

        #close enough to event
        if close: return True, True, 1.0

        #half the stepsize to creep closer to event (bisection)
        return result, False, 0.5


    def resolve(self, t):
        """Resolve the event and record the time (t) at which it occurs. 
        Resolves event using the action function (func_act) if it is defined. 

        Deactivates the event tracking upon first resolution.

        Parameters
        ----------
        t : float
            evaluation time for event resolution 
        """

        #save the time of event resolution
        self._times.append(t)

        #action function for event resolution
        if self.func_act is not None:
            self.func_act(t)

        #deactivate condition tracking
        self.off()