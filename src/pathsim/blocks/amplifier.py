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
        _out, self.outputs[0] = self.outputs[0], self.op_alg(self.inputs[0])
        return abs(_out - self.outputs[0])
