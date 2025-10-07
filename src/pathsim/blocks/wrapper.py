#########################################################################################
##
##                                    WRAPPER BLOCK 
##                             (pathsim/blocks/wrapper.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block
from ..events import Schedule


# BLOCK DEFINITIONS =====================================================================

class Wrapper(Block):
    """Wrapper block for discrete implementation and external code integration.

    The `Wrapper` class is designed to call the internal `func` at fixed intervals 
    using an internal `Schedule` event. This makes it particularly useful for wrapping 
    external code or implementing discrete-time systems.

    Essentially this block does the same as `Function` with the difference that its 
    not evaluated continuously but periodically at discrete times.


    Example
    -------
    There are two ways to setup the `Wrapper`, first and standard way is to define 
    a function to be wrapped and pass it to the block initializer:

    .. code-block:: python
        
        from pathsim.blocks import Wrapper
        
        #function to be wrapped
        def func(a, b, c):
            return a * (b + c)

        wrp = Wrapper(func, T=0.1)


    Another option is to use the `dec` classmethod, which might be more convenient 
    in some situations:

    .. code-block:: python
        
        from pathsim.blocks import Wrapper
        
        @Wrapper.dec(T=0.1)
        def wrp(a, b, c):
            return a * (b + c)


    This way the internal function of the block `wrp` will be evaluated with a period 
    of `T=0.1` and its outputs updated accordingly.


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
    Evt : Schedule
        internal event. Used for periodic sampling the wrapped method
    """

    def __init__(self, func=None, T=1, tau=0):
        super().__init__()
        self._T   = T
        self._tau = tau

        #assign func to wrap (direct initialization)
        if callable(func):
            self.func = func

        def _sample(t):

            #read current inputs
            u = self.inputs.to_array()

            #compute operator output
            y = self.func(*u)

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
        """decorator for direct instance construction from func
    
        Example
        -------
        Decorate a function definition to directly make it 
        a `Wrapper` block instance:

        .. code-block:: python
        
        from pathsim.blocks import Wrapper
        
        @Wrapper.dec(T=0.1)
        def wrp(a, b, c):
            return a * (b + c)


        Parameters
        ----------
        tau : float
            delay time for the start time of the wrapper sampling
        T : float
            sampling period for calling the `wrapped` function
        """
        def decorator(func):
            return cls(func, T, tau)
        return decorator
