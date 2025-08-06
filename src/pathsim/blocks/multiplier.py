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
    #max number of ports
    _n_in_max = None
    _n_out_max = 1

    #maps for input and output port labels
    _port_map_out = {"out": 0}

    def __init__(self):
        super().__init__()

        self.op_alg = Operator(
            func=prod, 
            jac=lambda x: np.array([
                prod(np.delete(x, i)) for i in range(len(x))
                ])
            )


    def update(self, t):
        """update system equation

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        self.outputs.update_from_array(self.op_alg(u))