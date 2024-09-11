#########################################################################################
##
##                            SOURCE BLOCKS (blocks/sources.py)
##
##           This module defines blocks that serve purely as inputs / sources 
##                for the simulation such as the generic 'Source' block
##
##                                 Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import (
    dict_to_array, 
    array_to_dict
    )



# INPUT BLOCKS ==========================================================================

class Constant(Block):
    """
    produces a constant output signal (SISO)
    (same as 'Source' with func=lambda t:value, 
    therefore one could argue that it is redundant)

    INPUTS : 
        value : (float) constant defining block output
    """

    def __init__(self, value=1):
        super().__init__()
        self.value = value

        #set output with value (DC)
        self.outputs[0] = self.value


    def reset(self):
        pass        



class Source(Block):
    """
    Generator, or source that produces an arbitrary time 
    dependent output, defined by the func (callable).

    INPUTS : 
        func : (callable) function defining time dependent block output
    """

    def __init__(self, func=lambda t: 1):
        super().__init__()

        if not callable(func):
            raise ValueError(f"'{func}' is not callable")

        self.func = func


    def update(self, t):
        #set output with internal function definition at time (t)
        self.outputs[0] = self.func(t)
        return 0.0
        