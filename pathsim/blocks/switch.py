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


    def __len__(self):
        """Algebraic passthrough only possible if state is defined"""
        return 0 if self.state is None else 1


    def select(self, state=0):
        """
        This method is unique to the `Switch` block and intended 
        to be used from outside the simulation level for selecting 
        the input ports for the switch state.
    
        This can be achieved for example with the event management 
        system and its callback/action functions.

        Parameters
        ---------
        state : int, None
            switch state / input port selection
        """
        self.state = state


    def update(self, t):
        """Update switch output depending on inputs

        Parameters
        ----------
        t : float
            evaluation time
        """

        self.outputs[0] = self.inputs.get(self.state, 0.0)
        return 0.0