#########################################################################################
##
##                       WRAPPER BLOCK (wrapper.py)
##
##                                2025
##
#########################################################################################

# IMPORTS ===============================================================================


from ._block import Block
from ..events import Schedule
from ..optim.operator import Operator


class Wrapper(Block):
    """
    Wrapper block for discrete implementation and external code integration.

    The `Wrapper` class is designed to trigger the `wrapped` method at fixed intervals 
    using an internal scheduled event. This makes it particularly useful for wrapping 
    external code or implementing discrete-time systems within the simulation framework.

    The block uses the `Schedule` class to periodically call the `_run_wrapper` method, 
    which must be implemented by subclasses. The inputs and outputs of the block are 
    handled through the `inputs` and `outputs` registers, enabling seamless integration 
    with other blocks in the simulation. 
    ...

    Parameters
    ----------
    func : callable
        function that defines algebraic block IO behaviour
    T : float
        sampling period for the wrapped function
    tau : float
        delay time for the start time of the event
        
    Attributes
    ----------
    Evt : 
        internal event. Used for periodic sampling the wrapped method
    tau : float
        delau time for the start time of the event
    T : float
        sampling period for the event
    """

    def __init__(self, func=None, T=1, tau=0):
        super().__init__()
        self._T   = T
        self._tau = tau

        #assign func to wrap (direct initialization)
        if callable(func):
            self.wrapped = func

        def _sample(t):

            #read current inputs
            u = self.inputs.to_array()

            #compute operator output
            y = self.wrapped(*u)

            #update block outputs
            self.outputs.update_from_array(y)

        #internal scheduled events
        self.Evt = Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                )
        self.events = [self.Evt]


    def update(self, t):
        """Update system equation for fixed point loop.
    
        Note
        ----
        No direct passthrough, the `Wrapper` block doesnt 
        implement the `update` method. The behavior is 
        defined by the `func` arg.

        Parameters
        ----------
        t : float
            evaluation time
        """
        pass
        
    
    @property
    def tau(self):
        """Getter for tau

        Returns
        -------
        tau : float
              delay time for the Schedule event
        """
        return self._tau

    @tau.setter
    def tau(self, value):
        """Setter for tau

        Parameters
        ----------
        value : float
                delay time
        """
        if value < 0:
            raise ValueError("tau must be non-negative")
        self._tau = value
        self.Evt.t_start = value

    @property
    def T(self):
        """Get the sampling period of the block
            
            Returns
            -------
            T: float
                sampling period for the Schedule event
        """
        return self._T

    @T.setter
    def T(self, value):
        """Set the sampling period of the block
            Parameters
            ----------
            value : float
                    sampling period
        """
        if value <= 0:
            raise ValueError("T must be positive")
        self._T = value
        self.Evt.t_period = value
    
    @classmethod
    def dec(cls, T=1, tau=0):
        """ decorator class for direct instance access from func"""
        def decorator(func):
            return cls(func, T, tau)
        return decorator
