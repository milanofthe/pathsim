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

from collections import defaultdict

from .connection import Connection

from .blocks._block import Block

from .utils.graph import Graph
from .utils.register import Register
from .utils.portreference import PortReference


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
    def __len__(self):
        return 0
    


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


    Example
    -------
    
    This is how we can wrap up multiple blocks within a subsystem. 
    In this case vanderpol system built from discrete components 
    instead of using an ODE block (in practice you should use 
    a monolithic ODE whenever possible due to performance).

    .. code-block:: python
        
        from pathsim import Subsystem, Interface, Connection
        from pathsim.blocks import Integrator, Function

        #van der Pol parameter
        mu = 1000

        #blocks in the subsystem
        If = Interface() # this is the interface to the outside
        I1 = Integrator(2)
        I2 = Integrator(0)
        Fn = Function(lambda x1, x2: mu*(1 - x1**2)*x2 - x1)

        sub_blocks = [If, I1, I2, Fn]

        #connections in the subsystem
        sub_connections = [
            Connection(I2, I1, Fn[1], If[1]), 
            Connection(I1, Fn, If), 
            Connection(Fn, I2)
            ]

        #the subsystem acts just like a normal block
        vdp = Subsystem(sub_blocks, sub_connections)


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

        #internal integration engine as 'None'
        self.engine = None

        #flag to set block (subsystem) active
        self._active = True

        #internal discrete events (for mixed signal blocks)
        self.events = []

        #operators for algebraic and dynamic components (not here)
        self.op_alg = None
        self.op_dyn = None

        #internal graph representation
        self.graph = None

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

        #assemble internal graph
        self._assemble_graph()


    def __len__(self):
        """Recursively compute the longest signal path in the subsytem by 
        depth first search, leveraging the '__len__' methods of the blocks. 

        This enables the path length computation even for nested subsystems.

        Iterate internal blocks and compute longest path from each block 
        as starting block.

        Basically the same as in the 'Simulation' class.
        """

        #no graph yet -> no passthrough anyway
        if not self.graph:
            return 0

        #internal loops -> tainted (inf)
        if self.graph.has_loops:
            return None

        #check if algebraic path from interface back to itself 
        is_alg = self.graph.is_algebraic_path(self.interface, self.interface)
        return int(is_alg)


    def __call__(self):
        """Recursively get the subsystems internal states of engines 
        (if available) of all internal blocks and nested subsystems 
        and the subsystem inputs and outputs as arrays for use outside. 

        Either for monitoring, postprocessing or event detection. 
        In any case this enables easy access to the current block state.
        """
        _inputs  = self.interface.outputs.to_array()
        _outputs = self.interface.inputs.to_array()
        _states  = []
        for block in self.blocks:
            _i, _o, _s = block()
            _states.append(_s)
        return _inputs, _outputs, np.hstack(_states)


    def __contains__(self, other):
        """Check if blocks and connections are already part of the subsystem

        Paramters
        ---------
        other : obj
            object to check if its part of subsystem

        Returns
        -------
        bool
        """
        return other in self.blocks or other in self.connections


    # methods for verification ----------------------------------------------------------

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


    # subsystem graph assembly --------------------------------------------------------------

    def _assemble_graph(self):
        """Assemble internal graph of subsystem for fast 
        algebraic evaluation during simulation.
        """
        self.graph = Graph(self.blocks, self.connections)


    # methods for access to metadata --------------------------------------------------------

    def size(self):
        """Get size information from subsystem, recursively assembled 
        from internal blocks, including nested subsystems.

        Returns
        -------
        sizes : tuple[int]
            size of block (default 1) and number 
            of internal states (from internal engine)
        """
        total_n, total_nx = 0, 0
        for block in self.blocks:
            n, nx = block.size()
            total_n += n
            total_nx += nx
        return total_n, total_nx


    # visualization -------------------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Plot the simulation results by calling all the blocks 
        that have visualization capabilities such as the 'Scope' 
        and 'Spectrum'.

        Parameters
        ----------
        args : tuple
            args for the plot methods
        kwargs : dict
            kwargs for the plot method
        """
        for block in self.blocks:
            block.plot(*args, **kwargs)


    # system management ---------------------------------------------------------------------

    def reset(self):
        """Reset the subsystem interface and all internal blocks"""

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


    def linearize(self, t):
        """Linearize the algebraic and dynamic components of the internal blocks.

        This is done by linearizing the internal 'Operator' and 'DynamicOperator' 
        instances of all the internal blocks of the subsystem in the current system 
        operating point. The operators create 1st order tayler approximations 
        internally and use them on subsequent calls after linarization.
    
        Recursively traverses down the hierarchy for nested subsystems and linearizes 
        all of them.

        Parameters
        ----------
        t : float 
            evaluation time
        """
        for block in self.blocks: 
            block.linearize(t)


    def delinearize(self):
        """Revert the linearization of the internal blocks."""
        for block in self.blocks: 
            block.delinearize()


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
            source_ports = conn_data["source"]["ports"]
            source = PortReference(source_block, source_ports)
            
            #target data
            targets = []
            for trg in conn_data["targets"]:
                target_block = id_to_block[trg["block"]]
                target_ports = trg["ports"]
                targets.append(
                    PortReference(target_block, target_ports)
                    )
            
            #create the connection
            connections.append(
                Connection(source, *targets)
                )
        
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

    @property    
    def inputs(self):
        return self.interface.outputs

    @property
    def outputs(self):
        return self.interface.inputs


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

        Collect convergence errors of internal blocks for algebraic 
        loop resolution.

        Parameters
        ----------
        t : float
            evaluation time 

        Returns
        ------- 
        max_error : float
            max deviation to previous iteration at evaluation 
            of alg. components
        """

        #update interface outgoing connections
        for connection in self.graph.outgoing_connections(self.interface):
            if connection: connection.update()

        #perform gauss-seidel iterations without error checking
        for _, blocks_dag, connections_dag in self.graph.dag():

            #update blocks at algebraic depth
            for block in blocks_dag:
                if block: block.update(t)

            #update connenctions at algebraic depth (data transfer)
            for connection in connections_dag:
                if connection: connection.update()

        #no internal algebraic loops -> early exit
        if not self.graph.has_loops:
            return 0.0

        #iterate DAG depths of broken loops
        max_error = 0.0
        for _, blocks_loop, connections_loop in self.graph.loop():

            #update blocks at algebraic depth
            for block in blocks_loop:

                #skip incative blocks
                if not block: 
                    continue

                #update block with error control enabled
                err = block.update(t)
                if err > max_error:
                    max_error = err

            #update connenctions at algebraic depth (data transfer)
            for connection in connections_loop:
                if connection: connection.update() 

        #return subsystem convergence error
        return max_error


    # methods for blocks with integration engines -------------------------------------------

    def solve(self, t, dt):
        """Advance solution of implicit update equation 
        for internal blocks.

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
        for block in self._blocks_dyn:
            if not block: continue
            err = block.solve(t, dt)
            if err > max_error:
                max_error = err
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
        for block in self._blocks_dyn:

            #skip inactive internal blocks
            if not block: continue

            suc, err_norm, scl = block.step(t, dt)
            
            #check solver stepping success
            if not suc: 
                success = False

            #update error tracking
            if err_norm > max_error_norm: 
                max_error_norm = err_norm
            
            #update timestep rescale if relevant
            if scl != 1.0 and scl > 0.0: 
                relevant_scales.append(scl)

        #no relevant timestep rescale -> quit early
        if not relevant_scales: 
            return success, max_error_norm, 1.0

        #compute real timestep rescale
        return success, max_error_norm, min(relevant_scales)


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

        #set internal dummy engine
        self.engine = Solver()

        #set integration engines and assemble list of dynamic blocks
        self._blocks_dyn = []
        for block in self.blocks:
            block.set_solver(Solver, **solver_args)
            if block.engine:
                self._blocks_dyn.append(block)


    def revert(self):
        """revert the internal blocks to the state 
        of the previous timestep 
        """
        for block in self._blocks_dyn:
            if block: block.revert()


    def buffer(self, dt):
        """buffer internal states of blocks with 
        internal integration engines 
        
        Parameters
        ----------
        dt : float
            evaluation time for buffering    
        """
        for block in self._blocks_dyn:
            if block: block.buffer(dt)