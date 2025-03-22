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


from ..optim.operator import Operator


# MIMO BLOCKS ===========================================================================

class Function(Block):
    """Arbitrary MIMO function block, defined by a callable object, 
    i.e. function or lambda expression.

    The function can have multiple arguments that are then provided 
    by the input channels of the function block.

    Form multi input, the function has to specify multiple arguments
    and for multi output, the aoutputs have to be provided as a 
    tuple or list. 

    Note
    -----
    If the outputs are provided as a single numpy array, they are 
    considered a single output. For MIMO, output has to be tuple.
    
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

    Parameters
    ---------- 
    func : callable
        MIMO function that defines block IO behaviour

    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator that wraps 'func'
    
    """

    def __init__(self, func=lambda x: x):
        super().__init__()

        #some checks to ensure that function works correctly
        if not callable(func):  
            raise ValueError(f"'{func}' is not callable")
        
        #function defining the block update
        self.func = func
        self.op_alg = Operator(func=lambda x: func(*x))


    def update(self, t):
        """Evaluate function block

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
                
        #apply operator to get output
        y = self.op_alg(dict_to_array(self.inputs))

        #set outputs to new values
        _outputs, self.outputs = self.outputs, array_to_dict(y)
        return max_error_dicts(_outputs, self.outputs)