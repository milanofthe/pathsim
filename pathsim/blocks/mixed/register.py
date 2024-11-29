#########################################################################################
##
##                              SAMPLE AND HOLD BLOCK
##                           (blocks/mixed/register.py)
##
##                               Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .._block import Block
from ...events.schedule import Schedule


# MIXED SIGNAL BLOCKS ===================================================================

class Register(Block):

    def __init__(self, size=3, T=1, tau=0):
        super().__init__()

        self.size = size
        self.T    = T
        self.tau  = tau

        #register output
        self.outputs = {i:0.0 for i in range(size)}

        #counter
        self._counter = 0

        def _sample(blocks, t):
            
            b = blocks[0]  

            b.outputs[b._counter] = b.inputs[0]
            b.outputs[b._counter+1] = 1

            #increment ring counter and reset
            b._counter = (b._counter + 1) % b.size

            if b._counter == 0: 
                b.outputs = {i:0.0 for i in range(b.size)}  
                b.outputs[0] = 1

        #internal scheduled events
        self.events = [
            Schedule(
                blocks=[self],
                t_start=tau,
                t_period=T,
                func_act=_sample
                ),
            ]