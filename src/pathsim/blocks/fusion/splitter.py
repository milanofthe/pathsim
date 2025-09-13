#########################################################################################
##
##                       GENERIC MIMO FUNCTION BLOCK (blocks/function.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ..function import Function


# BLOCKS ================================================================================

class Splitter(Function):
    """Splitter block that splits the input signal into multiple 
    outputs based on specified fractions.

    Note
    ----
    The output fractions must sum to one.
    
    Parameters
    ----------
        fractions : np.ndarray | list
    """

    #max number of ports
    _n_in_max = 1

    #maps for input and output port labels
    _port_map_in = {"in": 0}
    
    def __init__(self, fractions=None):

        self.fractions = np.ones(1) if fractions is None else np.array(fractions)

        #input validation
        if not np.isclose(sum(self.fractions), 1):
            raise ValueError(f"'fractions' must sum to one and not {sum(self.fractions)}")

        #dynamically define output port map based on fractions
        _port_map_out = {f"out {fr}": i for i, fr in enumerate(self.fractions)}

        #initialize like `Function` block
        super().__init__(func=lambda u: self.fractions*u)