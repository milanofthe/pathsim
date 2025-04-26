#########################################################################################
##
##                                 AMPLIFIER BLOCK 
##                              (blocks/amplifier.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block

from ..optim.operator import Operator


# SISO BLOCKS ===========================================================================

class Amplifier(Block):
    """Amplifies the input signal by 
    multiplication with a constant gain term like this:

    .. math::
        
        y(t) = \\mathrm{gain} \\cdot u(t)

    
    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.

        
    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #amplification by factor 5
        A = Amplifier(gain=5)


    Parameters
    ----------
    gain : float
        amplifier gain

        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

        self.op_alg = Operator(
            func=lambda x: x*self.gain, 
            jac=lambda x: self.gain
            )


    def update(self, t):
        """update system equation in fixed point loop

        Note
        ----
        SISO block has optimized 'update' method

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            deviation to previous iteration for convergence control
        """
        y = self.op_alg(self.inputs[0])
        return self.outputs.update_from_array_max_err(y)
