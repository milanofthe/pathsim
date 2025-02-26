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


# MISO BLOCKS ===========================================================================

class Adder(Block):
    """Summs / adds all input signals (MISO)"""

    def update(self, t):
        """update system equation in fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            relative error to previous iteration for convergence control
        """
        prev_output = self.outputs[0]
        self.outputs[0] = sum(v for v in self.inputs.values())
        return abs(prev_output - self.outputs[0])
