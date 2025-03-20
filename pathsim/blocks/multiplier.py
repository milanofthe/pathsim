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

from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Multiplier(Block):
    """multiplies all input signals (MISO)
      
    .. math::
        
        y(t) = \\prod_i u_i(t)

    """
    def __init__(self):
        super().__init__()

        self.op_alg = Operator(
            func=lambda x: prod(x), 
            jac=lambda x: np.array([prod(x[:i]+x[i+1:]) for i in range(len(x))])
            )


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
        _out, self.outputs[0] = self.outputs[0], self.op_alg(self.inputs.values())
        return abs(_out - self.outputs[0])
