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
        
    Notes
    -----
    same as 'Source' with func=lambda t:value, 
    therefore one could argue that it is redundant

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
            relative error to previous iteration for convergence control
        """
        self.outputs[0] = self.value
        return 0.0


class Source(Block):
    """Source that produces an arbitrary time dependent output, 
    defined by the func (callable).

    Example
    -------

    For example a ramp:
    ```pyhon
    S = Source(lambda t : t)
    ```
    
    or a simple sinusoid 
    ```python
    import numpy as np
    S = Source(np.sin)
    ```

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
        