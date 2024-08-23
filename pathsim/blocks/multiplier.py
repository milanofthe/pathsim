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

from ._block import Block

from ..utils.funcs import dict_to_array


# MISO BLOCKS ===========================================================================

class Multiplier(Block):
    """
    multiplies / product of all input signals (MISO)
    """

    def update(self, t):
        prev_output = self.outputs[0]
        self.outputs[0] = np.prod(dict_to_array(self.inputs), axis=0)
        return abs(prev_output - self.outputs[0])
