#########################################################################################
##
##                                      ADDER BLOCK 
##                                   (blocks/adder.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.utils import dict_to_array


from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Adder(Block):
    """Summs / adds all input signals (MISO)
    
    .. math::
        
        y(t) = \\sum_i u_i(t)

    """
    def __init__(self):
        super().__init__()

        self.op_alg = Operator(
            func=lambda x: sum(x), 
            jac=lambda x: np.ones_like(x)
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
