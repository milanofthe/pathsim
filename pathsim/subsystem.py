#########################################################################################
##
##                                 SUBSYSTEM DEFINITION 
##                                    (subsystem.py)
##
##              This module contains the 'Subsystem' and 'Interface' classes 
##         that manage subsystems that can be embedded within a larger simulation
##
##                                  Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

from .blocks._block import Block
from .utils.funcs import path_length_dfs


# IO CLASS ==============================================================================

class Interface(Block):
    """
    Bare-bone block that serves as a data interface for the 'Subsystem' class.

    It works like this:
    - Internal blocks of the subsystem are connected to the inputs and outputs 
      of this Interface block via the internal connections.
    - It behaves like a normal block (inherits the main 'Block' class methods).
    - It implements some special methods to get and set the inputs and outputs 
      of the blocks, that are used to translate between the internal blocks of the 
      subsystem and the inputs and outputs of the subsystem.
    - Handles data transfer to and from the internal subsystem blocks 
      to and from the inputs and outputs of the subsystem.
    """
    
    def set_output(self, port, value): 
        self.outputs[port] = value
    
    def get_input(self, port): 
        return self.inputs.get(port, 0.0)


# MAIN SUBSYSTEM CLASS ==================================================================

class Subsystem(Block):
    """
    Subsystem class that holds its own blocks and connecions and 
    can natively interface with the main simulation loop. 

    IO interface is realized by a special 'Interface' block, that has extra 
    methods for setting and getting inputs and outputs and serves 
    as the interface of the internal blocks to the outside. 

    The subsystem doesnt use its 'inputs' and 'outputs' dicts directly. 
    It exclusively handles data transfer via the 'Interface' block. 

    This class can be used just like any other block during the simulation, 
    since it implements the required methods 'update' for the fixed-point 
    iteration (resolving algebraic loops with instant time blocks), 
    the 'step' method that performs timestepping (especially for dynamic 
    blocks with internal states) and the 'solve' method for solving the 
    implicit update equation for implicit solvers. 

    INPUTS : 
        blocks      : (list of 'Block' objects) internal blocks of the subsystem
        connections : (list of 'Connection' objects) internal connections of the subsystem
    """

    def __init__(self, blocks=None, connections=None):
        super().__init__()

        #internal connecions
        self.connections = [] if connections is None else connections
        
        #collect and organize internal blocks
        self.blocks, self.interface = [], None

        if blocks is not None:
            for block in blocks:
                if isinstance(block, Interface): self.interface = block
                else: self.blocks.append(block)

        #check if interface is defined
        if self.interface is None:
            raise ValueError("Subsystem 'blocks' list needs to contain 'Interface' block!")

        #validate the internal connections upon initialization
        self._check_connections()


    def __len__(self):
        """
        Recursively compute the longest signal path in the subsytem by 
        depth first search, leveraging the '__len__' methods of the blocks. 
        This enables the path length computation even for nested subsystems.

        Iterate internal blocks and compute longest path from each block 
        as starting block.

        Basically the same as in the 'Simulation' class.
        """
        max_path_length = 0
        for block in self.blocks:
            path_length = path_length_dfs(self.connections, block)
            if path_length > max_path_length:
                max_path_length = path_length
        return max_path_length


    def _check_connections(self):
        """
        Check if connections are valid and if there is no input port that recieves 
        multiple outputs and could be overwritten unintentionally.

        If multiple outputs are assigned to the same input, an error is raised.
        """

        #iterate connections and check if they are valid
        for i, conn_1 in enumerate(self.connections):

            #check if connections overwrite each other and raise exception
            for conn_2 in self.connections[(i+1):]:
                if conn_1.overwrites(conn_2):
                    _msg = f"{conn_1} overwrites {conn_2}"
                    raise ValueError(_msg)


    def reset(self):

        #reset interface
        self.interface.reset()

        #reset internal blocks
        for block in self.blocks:
            block.reset()


    # methods for inter-block data transfer -------------------------------------------------

    def set(self, port, value):
        """
        The 'set' method of the 'Subsystem' sets the output 
        values of the 'Interface' block.
        """
        self.interface.set_output(port, value)


    def get(self, port):
        """
        The 'get' method of the 'Subsystem' retrieves the input 
        values of the 'Interface' block.
        """
        return self.interface.get_input(port)


    # methods for data recording ------------------------------------------------------------

    def sample(self, t):

        """
        Update the internal connections again and sample data from 
        the internal blocks that implement the 'sample' method.
        """

        #update internal connenctions (data transfer)
        for connection in self.connections:
            connection.update()

        #record data if required
        for block in self.blocks:
            block.sample(t)


    # methods for block output and state updates --------------------------------------------

    def update(self, t):
        """
        Update the instant time components of the internal blocks 
        to evaluate the (distributed) system equation.
        """

        #update internal connections (data transfer)
        for connection in self.connections:
            connection.update()

        #update internal blocks
        max_error = 0.0
        for block in self.blocks:
            error = block.update(t)
            if error > max_error:
                max_error = error

        #return subsystem convergence error
        return max_error


    def solve(self, t, dt):
        """
        Advance solution of implicit update equation for internal blocks.
        """
        max_error = 0.0
        for block in self.blocks:
            error = block.solve(t, dt)
            if error > max_error:
                max_error = error
        return max_error


    def step(self, t, dt):
        """
        Explicit component of timestep for internal blocks including error propagation.
        """

        #initial timestep rescale and error estimate
        success, max_error_abs, max_error_rel, relevant_scales = True, 0.0, 0.0, []

        #step blocks and get error estimates if available
        for block in self.blocks:
            ss, error_abs, error_rel, scl = block.step(t, dt)
            if not ss: success = False
            if error_abs > max_error_abs: max_error_abs = error_abs
            if error_rel > max_error_rel: max_error_rel = error_rel
            if scl not in [0.0, 1.0]: relevant_scales.append(scl)

        #calculate real relevant timestep rescale
        scale = 1.0 if not relevant_scales else min(relevant_scales)

        return success, max_error_abs, max_error_rel, scale


    # methods for blocks with integration engines -------------------------------------------

    def set_solver(self, Solver, **solver_args):
        """
        Initialize all blocks with solver for numerical integration
        and additional args for the solver such as tolerances, etc.

        If blocks already have solvers, change the numerical integrator
        to the 'Solver' class.
        
        INPUTS:
            Solver : ('Solver' class) numerical solver definition
            tolerance_lte : (float) tolerance for local truncation error
        """

        #iterate all blocks and set integration engines
        for block in self.blocks:
            block.set_solver(Solver, **solver_args)


    def revert(self):
        """
        revert the internal blocks to the state 
        of the previous timestep 
        """
        for block in self.blocks:
            block.revert()


    def buffer(self):
        """
        buffer internal states of blocks with 
        internal integration engines 
        """
        for block in self.blocks:
            block.buffer()
