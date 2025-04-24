#########################################################################################
##
##                            SOURCE BLOCKS (blocks/sources.py)
##
##           This module defines blocks that serve purely as inputs / sources 
##                for the simulation such as the generic 'Source' block
##
##                                 Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block


# INPUT BLOCKS ==========================================================================

class Constant(Block):
    """Produces a constant output signal (SISO)
        
    Parameters
    ----------
    value : float
        constant defining block output
    """

    def __init__(self, value=1):
        super().__init__()
        self.value = value
        

    def update(self, t):
        """update system equation fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
        self.outputs[0] = self.value
        return 0.0


class Source(Block):
    """Source that produces an arbitrary time dependent output, 
    defined by the func (callable).

    .. math::
    
        y(t) = \\mathrm{func}(t)


    Note
    ----
    This block is purely algebraic and its internal function (`func`) will 
    be called multiple times per timestep, each time when `Simulation._update(t)` 
    is called in the global simulation loop.


    Example
    -------
    For example a ramp:

    .. code-block:: python

        from pathsim.blocks import Source

        src = Source(lambda t : t)
    
    or a simple sinusoid with some frequency:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import Source
    
        #some parameter
        omega = 100
    
        #the function that gets evaluated
        def f(t):
            return np.sin(omega * t)

        src = Source(f)


    Parameters
    ---------- 
    func : callable
        function defining time dependent block output
    """

    def __init__(self, func=lambda t: 1):
        super().__init__()

        if not callable(func):
            raise ValueError(f"'{func}' is not callable")

        self.func = func


    def update(self, t):
        """update system equation fixed point loop 
        by evaluating the internal function 'func'

        Note
        ----
        No direct passthrough, so the `update` method 
        is optimized and has no convergence check

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            relative error to previous iteration for convergence control
        """
        self.outputs[0] = self.func(t)
        return 0.0
        