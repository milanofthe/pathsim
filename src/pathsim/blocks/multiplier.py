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

import numpy as np

from math import prod

from ._block import Block
from ..utils.utils import dict_to_array
from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Multiplier(Block):
    """multiplies all input signals (MISO)
      
    .. math::
        
        y(t) = \\prod_i u_i(t)

    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator that wraps 'prod'
    """

    def __init__(self):
        super().__init__()

        self.op_alg = Operator(
            func=prod, 
            jac=lambda x: np.array([
                prod(np.delete(x, i)) for i in range(len(x))
                ])
            )


    def update(self, t):
        """update system equation in fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
        u = dict_to_array(self.inputs)
        _out, self.outputs[0] = self.outputs[0], self.op_alg(u)
        return abs(_out - self.outputs[0])
