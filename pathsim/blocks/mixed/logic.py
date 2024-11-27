#########################################################################################
##
##                                  LOGIC BLOCKS
##                             (blocks/mixed/logic.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .._block import Block
from ...utils.utils import dict_to_array


# MIXED SIGNAL BLOCKS ===================================================================

class And(Block):
    """
    Basic logic block that sets its output to '1' if all input 
    signals are larger then a predefined threshold, else '0'.
    """

    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold


    def update(self, t):
        for v in self.inputs.values():
            if v <= self.threshold:
                self.outputs[0] = 0
                return 0.0
        self.outputs[0] = 1
        return 0.0


class Or(Block):
    """
    Basic logic block that sets its output to '1' if at least one 
    input signal is larger then a predefined threshold, else '0'.
    """

    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold


    def update(self, t):
        for v in self.inputs.values():
            if v > self.threshold:
                self.outputs[0] = 1
                return 0.0
        self.outputs[0] = 0
        return 0.0


class Not(Block):
    """
    Basic logic block that sets the outputs to the corresponding 
    input to '1' if the input signal is below or equal to a certain 
    threshold, else sets them to '0'.
    """

    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold
        

    def update(self, t):
        for k, v in self.inputs.items():
            self.outputs[k] = 1 if self.threshold <= 0 else 0
        return 0.0