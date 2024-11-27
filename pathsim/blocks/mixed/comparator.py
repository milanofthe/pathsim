#########################################################################################
##
##                                COMPARATOR BLOCK
##                           (blocks/mixed/comparator.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .._block import Block
from ...events.zerocrossing import ZeroCrossingUp, ZeroCrossingDown


# MIXED SIGNAL BLOCKS ===================================================================

class Comparator(Block):
    """
    Comparator block that sets the output to '1' it the input 
    signal crosses a predefined threshold and to '0' if it 
    crosses in the reverse direction. 

    This is realized by the block spawning two zero-crossing 
    event detectors that watch the inputs of the block and 
    locate the transitions up to a tolerance. Their callbacks 
    set the blocks outputs accordingly. 
    """

    def __init__(self, threshold=0, tolerance=1e-4):
        super().__init__()

        self.threshold = threshold
        self.tolerance = tolerance

        def func_evt(blocks, t):
            return blocks[0].inputs[0] - self.threshold

        def func_act_up(blocks, t):
            blocks[0].outputs[0] = 1

        def func_act_down(blocks, t):
            blocks[0].outputs[0] = 0

        #internal scheduled events
        self.events = [
            ZeroCrossingUp(
                blocks=[self], 
                func_evt=func_evt, 
                func_act=func_act_up,
                tolerance=tolerance
                ),
            ZeroCrossingDown(
                blocks=[self], 
                func_evt=func_evt, 
                func_act=func_act_down,
                tolerance=tolerance
                )
            ]