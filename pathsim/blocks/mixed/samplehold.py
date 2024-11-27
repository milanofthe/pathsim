#########################################################################################
##
##                              SAMPLE AND HOLD BLOCK
##                           (blocks/mixed/samplehold.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .._block import Block
from ...events.schedule import Schedule


# MIXED SIGNAL BLOCKS ===================================================================

class SampleHold(Block):

    def __init__(self, T=1, tau=0):
        super().__init__()

        self.T   = T
        self.tau = tau

        def sample(blocks, t):
            blocks[0].outputs = blocks[0].inputs

        #internal scheduled events
        self.events = [
            Schedule(
                blocks=[self],
                t_start=tau,
                t_period=T,
                func_act=sample
                ),
            ]