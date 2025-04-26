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
from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Multiplier(Block):
    """Multiplies all signals from all input ports (MISO).
      
    .. math::
        
        y(t) = \\prod_i u_i(t)

            
    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.


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
        u = self.inputs.to_array()
        return self.outputs.update_from_array_max_err(self.op_alg(u))
