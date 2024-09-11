#########################################################################################
##
##                              IDEAL AMPLIFIER BLOCK 
##                              (blocks/amplifier.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block


# SISO BLOCKS ===========================================================================

class Amplifier(Block):
    """
    amplifies the input signal by 
    multiplication with a constant gain term 

    INPUTS : 
        gain : (float) amplifier gain
    """

    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain


    def update(self, t):
        prev_output = self.outputs[0]
        self.outputs[0]  = self.gain * self.inputs[0]
        return abs(prev_output - self.outputs[0])
