#########################################################################################
##
##                       GENERIC MIMO FUNCTION BLOCK (blocks/function.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.utils import (
    max_error_dicts, 
    array_to_dict,
    dict_to_array
    )


# MIMO BLOCKS ===========================================================================

class Function(Block):
    """Arbitrary MIMO function block, defined by a callable object, 
    i.e. function or lambda expression.

    The function can have multiple arguments that are then provided 
    by the input channels of the function block.

    Form multi input, the function has to specify multiple arguments
    and for multi output, the aoutputs have to be provided as a 
    tuple or list. 

    Parameters
    ---------- 
    func : callable
        MIMO function that defines block IO behaviour

    Notes
    -----
    If the outputs are provided as a single numpy array,
    they are considered a single output
    
    Example
    -------
    consider the function: 

    .. code-block:: python
    
        from pathsim.blocks import Function

        def f(a, b, c):
            return a**2, a*b, b/c

        fn = Function(f)
        

    then the input channels of the block are assigned 
    to the function arguments following this scheme:

    .. code-block::

        inputs[0] -> a
        inputs[1] -> b
        inputs[2] -> c

    and the function outputs are assigned to the 
    output channels of the block in the same way:

    .. code-block::

        a**2 -> outputs[0]
        a*b  -> outputs[1]
        b/c  -> outputs[2]
    
    """

    def __init__(self, func=lambda x: x):
        super().__init__()

        #some checks to ensure that function works correctly
        if not callable(func):  
            raise ValueError(f"'{func}' is not callable")
        
        #function defining the block update
        self.func = func


    def update(self, t):
        """update system equation fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            relative error to previous iteration for convergence control
        """

        #compute function output
        output = self.func(*dict_to_array(self.inputs))

        #check if the output is scalar
        if np.isscalar(output):
            prev_output = self.outputs[0]
            self.outputs[0] = output
            return abs(prev_output - self.outputs[0])
        else:
            prev_outputs = self.outputs.copy()
            self.outputs = array_to_dict(output)
            return max_error_dicts(prev_outputs, self.outputs)
