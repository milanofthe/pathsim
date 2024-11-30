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
    """
    summs / adds all input signals (MISO)
    """

    def update(self, t):
        prev_output = self.outputs[0]
        self.outputs[0] = sum(v for v in self.inputs.values())
        return abs(prev_output - self.outputs[0])
