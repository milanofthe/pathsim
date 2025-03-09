#########################################################################################
##
##                                   SWITCH BLOCK
##                                (blocks/switch.py)
##
##                                Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block


# BLOCK DEFINITION ======================================================================

class Switch(Block):
    """Switch block that selects between its inputs and copies 
    one of them to the output. 
    
    Sets block output depending on `self.state` like this:

        state = None -> outputs[0] = 0

        state = 0 -> outputs[0] = inputs[0]

        state = 1 -> outputs[0] = inputs[1]

        state = 2 -> outputs[0] = inputs[2]
    
        ...

    Parameters
    ----------
    state : int, None
        state of the switch
    
    """

    def __init__(self, state=None):
        super().__init__()

        self.state = state


    def update(self, t):
        """Update switch output depending on inputs

        Parameters
        ----------
        t : float
            evaluation time
        """

        self.outputs[0] = self.inputs.get(self.state)
        return 0.0