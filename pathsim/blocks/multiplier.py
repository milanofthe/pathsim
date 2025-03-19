#########################################################################################
##
##                        REDUCTION BLOCKS (blocks/multiplier.py)
##
##                     This module defines static 'Multiplier' block
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from math import prod

from ._block import Block


# MISO BLOCKS ===========================================================================

class Multiplier(Block):
    """multiplies all input signals (MISO)
      
    .. math::
        
        y(t) = \\prod_i u_i(t)

    """

    def _func_alg(self, x, u, t):
        return prod(u)


    def update(self, t):
        """update system equation in fixed point loop

        Note
        ----
        This is a MISO block with an optimized 'update' method for this case

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
        _out, self.outputs[0] = self.outputs[0], self._func_alg(0, self.inputs.values(), t)
        return abs(_out - self.outputs[0])
