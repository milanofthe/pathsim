#########################################################################################
##
##                          REDUCTION BLOCKS (blocks/adder.py)
##
##                       This module defines static 'Adder' block 
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import dict_to_array


# MISO BLOCKS ===========================================================================

class Adder(Block):
    """
    summs / adds all input signals (MISO)
    """

    def update(self, t):
        prev_output = self.outputs[0]
        self.outputs[0] = np.sum(dict_to_array(self.inputs), axis=0)
        return abs(prev_output - self.outputs[0])
