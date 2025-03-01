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

import numpy as np

from .blocks._block import Block
from .utils.utils import path_length_dfs, dict_to_array


# IO CLASS ==============================================================================

class Interface(Block):
    """Bare-bone block that serves as a data interface for the 'Subsystem' class.

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
    """Subsystem class that holds its own blocks and connecions and 
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

    Parameters
    ----------
    blocks : list[Block] 
        internal blocks of the subsystem
    connections : list[Connection]
        internal connections of the subsystem

    Attributes
    ----------
    interface : Interface
        internal interface block for data transfer to the outside
    """

    def __init__(self, blocks=None, connections=None):
        super().__init__()

        #internal connecions
        self.connections = [] if connections is None else connections
        
        #collect and organize internal blocks
        self.blocks, self.interface = [], None

        if blocks is not None:
            for block in blocks:
                if isinstance(block, Interface): 
                    
                    if self.interface is not None:
                        #interface block is already defined
                        raise ValueError("Subsystem can only have one 'Interface' block!")
                    
                    self.interface = block
                else: 
                    #regular blocks
                    self.blocks.append(block)

        #check if interface is defined
        if self.interface is None:
            raise ValueError("Subsystem 'blocks' list needs to contain 'Interface' block!")

        #validate the internal connections upon initialization
        self._check_connections()


    def __len__(self):
        """Recursively compute the longest signal path in the subsytem by 
        depth first search, leveraging the '__len__' methods of the blocks. 

        This enables the path length computation even for nested subsystems.

        Iterate internal blocks and compute longest path from each block 
        as starting block.

        Basically the same as in the 'Simulation' class.
        """
        max_path_length = 0
        for block in [self.interface, *self.blocks]:
            path_length = path_length_dfs(self.connections, block)
            if path_length > max_path_length:
                max_path_length = path_length
        return max_path_length


    def __call__(self):
        """Recursively get the subsystems internal states of engines 
        (if available) of all internal blocks and nested subsystems 
        and the subsystem inputs and outputs as arrays for use outside. 

        Either for monitoring, postprocessing or event detection. 
        In any case this enables easy access to the current block state.
        """
        _inputs  = dict_to_array(self.interface.outputs)
        _outputs = dict_to_array(self.interface.inputs)
        _states  = []
        for block in self.blocks:
            _i, _o, _s = block()
            _states.append(_s)
        return _inputs, _outputs, np.hstack(_states)


    def _check_connections(self):
        """Check if connections are valid and if there is no input port 
        that recieves multiple outputs and could be overwritten unintentionally.

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
        """Reset the subsystem and all internal blocks"""

        #reset interface
        self.interface.reset()

        #reset internal blocks
        for block in self.blocks:
            block.reset()


    def on(self):
        """Activate the subsystem and all internal blocks, sets the boolean
        evaluation flag to 'True'.
        """
        self._active = True
        for block in self.blocks: 
            block.on()
    

    def off(self):
        """Deactivate the subsystem and all internal blocks, sets the boolean
        evaluation flag to 'False'. Also resets the subsystem.
        """
        self._active = False
        for block in self.blocks: 
            block.off()
        self.reset()


    # serialization / deserialization -------------------------------------------------------
    
    def to_dict(self):
        """Custom serialization for Subsystem"""
        data = super().to_dict()
        
        #serialization for internal blocks and interface
        data["params"]["blocks"] = [block.to_dict() for block in self.blocks + [self.interface]]

        #serialize connections
        data["params"]["connections"] = [conn.to_dict() for conn in self.connections]
        
        return data

    
    @classmethod
    def from_dict(cls, data):
        """Custom deserialization for Subsystem"""
        from .connection import Connection
        
        #deserialize blocks and create block ID mapping
        blocks, id_to_block = [], {}
        for blk_data in data["params"].pop("blocks", []):
            block = Block.from_dict(blk_data)
            blocks.append(block)
            id_to_block[blk_data["id"]] = block

        #deserialize connections
        connections = []
        for conn_data in data["params"].pop("connections", []):

            #source data
            source_block = id_to_block[conn_data["source"]["block"]]
            source_port = conn_data["source"]["port"]
            
            #target data
            targets = []
            for trg in conn_data["targets"]:
                target_block = id_to_block[trg["block"]]
                target_port = trg["port"]
                targets.append((target_block, target_port))
            
            #create the connection
            connections.append(Connection((source_block, source_port), *targets))
        
        #finally construct the subsystem
        return cls(blocks, connections)
        

    # methods for discrete event management -------------------------------------------------

    def get_events(self):
        """Recursively collect and return events spawned by the 
        internal blocks of the subsystem, for discrete time 
        blocks such as triggers / comparators, clocks, etc.
        """
        _events = []
        for block in self.blocks:
            _events.extend(block.get_events())
        return _events


    # methods for inter-block data transfer -------------------------------------------------

    def set(self, port, value):
        """The 'set' method of the 'Subsystem' sets the output 
        values of the 'Interface' block.
    
        Parameters
        ----------
        port : int
            input port to set value to
        value : numeric
            value to set at input port (of subsystem)
        """
        self.interface.set_output(port, value)


    def get(self, port):
        """The 'get' method of the 'Subsystem' retrieves the input 
        values of the 'Interface' block.
    
        Parameters
        ----------
        port : int
            output port (of subsystem) to retrieve value from
        """
        return self.interface.get_input(port)


    # methods for data recording ------------------------------------------------------------

    def sample(self, t):
        """Update the internal connections again and sample data from 
        the internal blocks that implement the 'sample' method.
    
        Parameters
        ----------
        t : float
            evaluation time 
        """

        #record data if required
        for block in self.blocks:
            block.sample(t)


    # methods for block output and state updates --------------------------------------------

    def update(self, t):
        """Update the instant time components of the internal blocks 
        to evaluate the (distributed) system equation.

        Parameters
        ----------
        t : float
            evaluation time 

        Returns
        ------- 
        max_error : float
            error tolerance of system equation convergence
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

        Parameters
        ----------
        t : float
            evaluation time 
        dt : float
            timestep
    
        Returns
        -------
        max_error : float
            maximum error of implicit update equaiton
        """
        max_error = 0.0
        for block in self.blocks:
            error = block.solve(t, dt)
            if error > max_error:
                max_error = error
        return max_error


    def step(self, t, dt):
        """Explicit component of timestep for internal blocks 
        including error propagation.

        Notes
        ----- 
        This is pretty much an exact copy of the '_step' method 
        from the 'Simulation' class.

        Parameters
        ---------- 
        t : float
            evaluation time 
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        """

        #initial timestep rescale and error estimate
        success, max_error_norm, relevant_scales = True, 0.0, []

        #step blocks and get error estimates if available
        for block in self.blocks:
            ss, err_norm, scl = block.step(t, dt)
            
            #check solver stepping success
            if not ss: 
                success = False

            #update error tracking
            if err_norm > max_error_norm: 
                max_error_norm = err_norm
            
            #update timestep rescale if relevant
            if scl not in [0.0, 1.0]: 
                relevant_scales.append(scl)

        #no relevant timestep rescale -> quit early
        if not relevant_scales: 
            return success, max_error_norm, 1.0

        #compute real timestep rescale
        return success, max_error_norm, min(relevant_scales)


    # methods for blocks with integration engines -------------------------------------------

    def set_solver(self, Solver, **solver_args):
        """Initialize all blocks with solver for numerical integration
        and additional args for the solver such as tolerances, etc.

        If blocks already have solvers, change the numerical integrator
        to the 'Solver' class.
        
        Parameters
        ----------
        Solver : Solver
            numerical solver definition
        solver_args : dict
            args to initialize solver with 
        """

        #iterate all blocks and set integration engines
        for block in self.blocks:
            block.set_solver(Solver, **solver_args)


    def revert(self):
        """revert the internal blocks to the state 
        of the previous timestep 
        """
        for block in self.blocks:
            block.revert()


    def buffer(self, dt):
        """buffer internal states of blocks with 
        internal integration engines 
        
        Parameters
        ----------
        dt : float
            evaluation time for buffering    
        """
        for block in self.blocks:
            block.buffer(dt)