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

from ..optim.operator import Operator


# MIMO BLOCKS ===========================================================================

class Function(Block):
    """Arbitrary MIMO function block, defined by a callable object, 
    i.e. function or `lambda` expression.

    The function can have multiple arguments that are then provided 
    by the input channels of the function block.

    Form multi input, the function has to specify multiple arguments
    and for multi output, the aoutputs have to be provided as a 
    tuple or list. 

    In the context of the global system, this block implements algebraic 
    components of the global system ODE/DAE.

    .. math::

        \\vec{y} = \\mathrm{func}(\\vec{u})
    

    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.
    Therefore `func` must be purely algebraic and not introduce states, 
    delay, etc. For interfacing with external stateful APIs, use the 
    API block.


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
        

    then, when the block is uldated, the input channels of the block are 
    assigned to the function arguments following this scheme:

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

    Because the `Function` block only has a single argument, it can be 
    used to decorate a function and make it a `PathSim` block. This might 
    be handy in some cases to keep definitions concise and localized 
    in the code:

    .. code-block:: python

        from pathsim.blocks import Function

        #does the same as the definition above
            
        @Function
        def fn(a, b, c):
            return a**2, a*b, b/c

        #'fn' is now a PathSim block


    Parameters
    ---------- 
    func : callable
        MIMO function that defines algebraic block IO behaviour


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
        """Evaluate function block as part of algebraic component 
        of global system DAE. 

        With convergence control for algebraic loop solver.

        Parameters
        ----------
        t : float
            evaluation time
        """
                
        #apply operator to get output
        y = self.op_alg(self.inputs.to_array())
        self.outputs.update_from_array(y)
