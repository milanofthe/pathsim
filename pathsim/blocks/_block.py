#########################################################################################
##
##                                     BASE BLOCK 
##                                 (blocks/_block.py)
##
##            This module defines the base 'Block' class that is the parent 
##         to all other blocks and can serve as a base for new or custom blocks
##                                
##
##                                  Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from ..utils.utils import dict_to_array
from ..utils.serialization import Serializable


# BASE BLOCK CLASS ======================================================================

class Block(Serializable):
    """Base 'Block' object that defines the inputs, outputs and the connect method.

    Block interconnections are handeled via the io interface of the blocks. 
    It is realized by dicts for the 'inputs' and for the 'outputs', where 
    the key of the dict is the input/output channel and the corresponding 
    value is the input/output value. 

    The block can spawn discrete events that are handled by the main simulation 
    for triggers, discrete time blocks, etc.

    Notes
    -----
    This block is not intended to be used directly and serves as a base 
    class definition for other blocks to be inherited.

    Attributes
    ----------
    inputs : dict{int: float}
        input value register of block
    outputs : dict{int: float}
        output value register of block
    engine : None, Solver
        numerical integrator instance
    events : list[Event]
        list of internal events, for mixed signal blocks
    _active : bool
        flag that sets the block active or inactive   
    """

    def __init__(self):

        #dicts to hold input and output values
        self.inputs  = {0:0.0}  
        self.outputs = {0:0.0} 

        #initialize integration engine as 'None' by default
        self.engine = None

        #flag to set block active
        self._active = True

        #internal discrete events (for mixed signal blocks)
        self.events = []


    def __len__(self):
        """The '__len__' method of the block is used to compute the length of the 
        algebraic path of the block. 

        For instant time blocks or blocks with purely algenbraic components 
        (adders, amplifiers, etc.) it returns 1, otherwise (integrator, delay, etc.) 
        it returns 0.

        If the block is disabled '_active == False', it returns 0 as well, since
        this breaks the signal path.

        Returns
        -------
        len : int
            length of the algebraic path of the block
        """
        return 1 if self._active else 0


    def __getitem__(self, key):
        """The '__getitem__' method is intended to make connection creation more 
        convenient and therefore just returns the block itself and the key directly 
        after doing some basic checks.

        Parameters
        ----------
        key : int
            key of the port

        Returns
        -------
        pair : tuple[Block, int]
            tuple of the block itself and the key of the port
        """
        if not isinstance(key, int):
            raise ValueError(f"Port has to be of type 'int' but is '{type(key)}'!")
        return (self, key)


    def __call__(self):
        """The '__call__' method returns internal states of engine 
        (if available) and the block inputs and outputs as arrays for 
        use outside. 

        Either for monitoring, postprocessing or event detection. 
        In any case this enables easy access to the current block state.

        Returns
        -------
        inputs : array
            block input register
        outputs : array
            block output register
        states : array
            internal states of the block
        """
        _inputs  = dict_to_array(self.inputs)
        _outputs = dict_to_array(self.outputs)
        _states  = self.engine.get() if self.engine else []
        return _inputs, _outputs, _states


    def __bool__(self):
        return self._active


    # methods for simulation management -------------------------------------------------

    def on(self):
        """Activate the block and all internal events, sets the boolean
        evaluation flag to 'True'.
        """
        self._active = True
        for event in self.events: 
            event.on()


    def off(self):
        """Deactivate the block and all internal events, sets the boolean 
        evaluation flag to 'False'. Also resets the block.
        """
        self._active = False
        for event in self.events: 
            event.off()
        self.reset()  


    def reset(self):
        """Reset the blocks inputs and outputs and also its internal solver, 
        if the block has a solver instance.
        """
        #reset inputs and outputs while maintaining ports
        self.inputs  = {k:0.0 for k in sorted(self.inputs.keys())}  
        self.outputs = {k:0.0 for k in sorted(self.outputs.keys())}

        #reset engine if block has solver
        if self.engine: self.engine.reset()


    # methods for blocks with discrete events -------------------------------------------

    def get_events(self):
        """Return internal events of the block, for discrete time blocks 
        such as triggers / comparators, clocks, etc.

        Returns
        -------
        events : list[Event]
            internal events of the block
        """
        return self.events


    # methods for blocks with integration engines ---------------------------------------

    def set_solver(self, Solver, **solver_args):
        """Initialize the numerical integration engine with local truncation error 
        tolerance if required.

        If the block already has an integration engine, it is changed, 
        if it does not require an integration engine, this method just passes.

        Parameters
        ----------
        Solver : Solver
            numerical integrator
        solver_args : dict
            additional args for the solver
        """
        pass


    def revert(self):
        """Revert the block to the state of the previous timestep, if the 
        block has a solver instance indicated by the 'has_engine' flag.

        This is required for adaptive solvers to revert the state to the 
        previous timestep.
        """
        if self.engine: self.engine.revert()


    def buffer(self, dt):
        """
        Buffer current internal state of the block and the current timestep
        if the block has a solver instance (is stateful).

        This is required for multistage, multistep and adaptive integrators.

        Parameters
        ----------
        dt : float
            integration timestep
        """
        if self.engine: self.engine.buffer(dt)


    # methods for sampling data ---------------------------------------------------------
    
    def sample(self, t):
        """Samples the data of the blocks inputs or internal state when called. 

        This can record block parameters after a succesful timestep such as 
        for the 'Scope' and 'Delay' blocks but also for sampling from a random 
        distribution in the 'RNG' and the noise blocks.
        
        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        pass


    # methods for inter-block data transfer ---------------------------------------------

    def set(self, port, value):
        """Set the value of an input port of the block.

        Parameters
        ----------
        port : int
            input port number
        value : int, float, complex
            value to set at input register port
        """        
        self.inputs[port] = value


    def get(self, port):
        """Get the value of an output port of the block.
        
        Uses the 'get' method of 'outputs' dict with default value '0.0'.
        
        Parameters
        ----------
        port : int
            output port number

        Returns
        -------
        value : int, float, complex
            value of the output register port
        """
        return self.outputs.get(port, 0.0)


    # methods for block output and state updates ----------------------------------------

    def update(self, t):
        """The 'update' method is called iteratively for all blocks BEFORE the timestep 
        to resolve algebraic loops (fixed-point iteraion). 

        It is meant for instant time blocks (blocks that dont have a delay due to the 
        timestep, such as Amplifier, etc.) and updates the 'outputs' of the block 
        directly based on the 'inputs' and possibly internal states. 

        It computes and returns the relative difference between the new output and 
        the previous output (before the step) to track convergence of the fixed-point 
        iteration.

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            relative error to previous iteration for convergence control
        """
        return 0.0 


    def solve(self, t, dt):
        """The 'solve' method performes one iterative solution step that is required 
        to solve the implicit update equation of the solver if an implicit solver 
        (numerical integrator) is used.

        It returns the relative difference between the new updated solution 
        and the previous iteration of the solution to track convergence within 
        an outer loop.

        This only has to be implemented by blocks that have an internal 
        integration engine with an implicit solver.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        return 0.0 


    def step(self, t, dt):
        """The 'step' method is used in transient simulations and performs an action 
        (numeric integration timestep, recording data, etc.) based on the current 
        inputs and the current internal state. 

        It performes one timestep for the internal states. For instant time blocks, 
        the 'step' method does not has to be implemented specifically. 
        
        The method handles timestepping for dynamic blocks with internal states
        such as 'Integrator', 'StateSpace', etc.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """

        #by default no error estimate (error norm -> 0.0)
        return True, 0.0, 1.0