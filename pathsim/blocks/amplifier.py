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


# SISO BLOCKS ===========================================================================

class Amplifier(Block):
    """Amplifies the input signal by 
    multiplication with a constant gain term like this:

    .. math::
        
        y(t) = \\mathrm{gain} \\cdot u(t)

    
    Parameters
    ----------
    gain : float
        amplifier gain
    """

    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain


    def _func_alg(self, x, u, t):
        return self.gain * u


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
        _out, self.outputs[0] = self.outputs[0], self._func_alg(0, self.inputs[0], t)
        return abs(_out - self.outputs[0])
