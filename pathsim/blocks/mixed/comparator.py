#########################################################################################
##
##                                COMPARATOR BLOCK
##                           (blocks/mixed/comparator.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================


import numpy as np

from .._block import Block
from ...events.zerocrossing import ZeroCrossing


# MIXED SIGNAL BLOCKS ===================================================================

class Comparator(Block):
    """
    Comparator block that sets the output to '1' it the input 
    signal crosses a predefined threshold and to '-1' if it 
    crosses in the reverse direction. 

    This is realized by the block spawning a zero-crossing 
    event detector that watches the input of the block and 
    locates the transition up to a tolerance. 
    
    The block output is determined by a simple sign check in
    the 'update' method.
    """

    def __init__(self, threshold=0, tolerance=1e-4):
        super().__init__()

        self.threshold = threshold
        self.tolerance = tolerance

        def func_evt(blocks, t):
            return blocks[0].inputs[0] - self.threshold

        #internal event for transition detection
        self.events = [
            ZeroCrossing(
                blocks=[self], 
                func_evt=func_evt, 
                tolerance=tolerance
                )
            ]


    def update(self, t):
        self.outputs[0] = np.sign(self.inputs[0] - self.threshold)
        return 0.0