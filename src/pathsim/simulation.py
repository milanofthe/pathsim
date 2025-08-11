#########################################################################################
##
##                               MAIN SIMULATION ENGINE
##                                   (simulation.py)
##
##                This module contains the simulation class that manages
##            the blocks, connections, events and specific simulation methods.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

import json
import datetime
import logging

from collections import defaultdict

from pathsim import __version__

from ._constants import (
    SIM_TIMESTEP,
    SIM_TIMESTEP_MIN,
    SIM_TIMESTEP_MAX,
    SIM_TOLERANCE_FPI,
    SIM_ITERATIONS_MAX,
    LOG_ENABLE
    )

from .optim.booster import ConnectionBooster

from .utils.graph import Graph
from .utils.analysis import Timer
from .utils.portreference import PortReference
from .utils.progresstracker import ProgressTracker

from .solvers import SSPRK22, SteadyState

from .blocks._block import Block

from .events._event import Event

from .connection import Connection


# TRANSIENT SIMULATION CLASS ============================================================

class Simulation:
    """Class that performs transient analysis of the dynamical system, defined by the 
    blocks and connecions. It manages all the blocks and connections and the timestep update.

    The global system equation is evaluated by fixed point iteration, so the information from 
    each timestep gets distributed within the entire system and is available for all blocks at 
    all times.

    The minimum number of fixed-point iterations 'iterations_min' is set to 'None' by default 
    and then the length of the longest internal signal path (with passthrough) is used as the 
    estimate for minimum number of iterations needed for the information to reach all instant 
    time blocks in each timestep. Dont change this unless you know that the actual path is 
    shorter or something similar that prohibits instant time information flow. 

    Convergence check for the fixed-point iteration loop with 'tolerance_fpi' is based on 
    max absolute error (max-norm) to previous iteration and should not be touched.

    Multiple numerical integrators are implemented in the 'pathsim.solvers' module. 
    The default solver is a fixed timestep 2nd order Strong Stability Preserving Runge Kutta 
    (SSPRK22) method which is quite fast and has ok accuracy, especially if you are forced to 
    take small steps to cover the behaviour of forcing functions. Adaptive timestepping and 
    implicit integrators are also available.
    
    Manages an event handling system based on zero crossing detection. Uses 'Event' objects 
    to monitor solver states of stateful blocks and applys transformations on the state in 
    case an event is detected. 

    Example
    -------

    This is how to setup a simple system simulation using the 'Simulation' class:

    .. code-block:: python
        
        import numpy as np

        from pathsim import Simulation, Connection
        from pathsim.blocks import Source, Integrator, Scope

        src = Source(lambda t: np.cos(2*np.pi*t))
        itg = Integrator()
        sco = Scope(labels=["source", "integrator"])
        
        sim = Simulation(
            blocks=[src, itg, sco],
            connections=[
                Connection(src[0], itg[0], sco[0]),
                Connection(itg[0], sco[1])    
                ],
            dt=0.01
            )

        sim.run(4)
        sim.plot()

    Parameters
    ----------
    blocks : list[Block] 
        blocks that define the system
    connections : list[Connection] 
        connections that connect the blocks
    events : list[Event]
        list of event trackers (zero crossing detection, schedule, etc.)
    dt : float
        transient simulation timestep in time units, 
        default see ´SIM_TIMESTEP´ in ´_constants.py´
    dt_min : float
        lower bound for transient simulation timestep, 
        default see ´SIM_TIMESTEP_MIN´ in ´_constants.py´
    dt_max : float
        upper bound for transient simulation timestep, 
        default see ´SIM_TIMESTEP_MAX´ in ´_constants.py´
    Solver : Solver 
        ODE solver class for numerical integration from ´pathsim.solvers´,
        default is ´pathsim.solvers.ssprk22.SSPRK22´ (2nd order expl. Runge Kutta)
    tolerance_fpi : float
        absolute tolerance for convergence of algebraic loops 
        and internal optimizers of implicit ODE solvers, 
        default see ´SIM_TOLERANCE_FPI´ in ´_constants.py´
    iterations_max : int
        maximum allowed number of iterations for implicit ODE 
        solver optimizers and algebraic loop solver, 
        default see ´SIM_ITERATIONS_MAX´ in ´_constants.py´
    log : bool | string
        flag to enable logging, default see ´LOG_ENABLE´ in ´_constants.py´
        (alternatively a path to a log file can be specified)
    solver_kwargs : dict
        additional parameters for numerical solvers such as absolute 
        (´tolerance_lte_abs´) and relative (´tolerance_lte_rel´) tolerance, 
        defaults are defined in ´_constants.py´

    Attributes
    ----------
    time : float
        global simulation time, starting at ´0.0´
    graph : Graph
        internal graph representation for fast system funcion evluations 
        using DAG with algebraic depths
    boosters : None | list[ConnectionBooster]
        list of boosters (fixed point accelerators) that wrap algebraic 
        loop closing connections assembled from the system graph
    engine : Solver
        global integrator (ODE solver) instance serving as a dummy to 
        get attributes and access to intermediate evaluation stages
    logger : logging.Logger
        global simulation logger
    _needs_buffering : bool
        flag for buffering system state
    _blocks_dyn : list[Block]
        list of blocks with internal ´Solver´ instances (stateful) 
    """

    def __init__(
        self, 
        blocks=None, 
        connections=None, 
        events=None,
        dt=SIM_TIMESTEP, 
        dt_min=SIM_TIMESTEP_MIN, 
        dt_max=SIM_TIMESTEP_MAX, 
        Solver=SSPRK22, 
        tolerance_fpi=SIM_TOLERANCE_FPI, 
        iterations_max=SIM_ITERATIONS_MAX, 
        log=LOG_ENABLE,
        **solver_kwargs
        ):

        #system definition
        self.blocks      = []
        self.connections = []
        self.events      = []

        #simulation timestep and bounds
        self.dt     = dt
        self.dt_min = dt_min
        self.dt_max = dt_max

        #numerical integrator to be used (class definition)
        self.Solver = Solver

        #numerical integrator instance
        self.engine = Solver()

        #internal system graph -> initialized later
        self.graph = None

        #internal algebraic loop solvers -> initialized later
        self.boosters = None

        #error tolerance for fixed point loop and implicit solver
        self.tolerance_fpi = tolerance_fpi

        #additional solver parameters
        self.solver_kwargs = solver_kwargs

        #iterations for fixed-point loop
        self.iterations_max = iterations_max

        #enable logging flag
        self.log = log

        #initial simulation time
        self.time = 0.0

        #flag for state buffering (transient)
        self._needs_buffering = True

        #collection of blocks with internal ODE solvers
        self._blocks_dyn = []

        #initialize logging for logging mode
        self._initialize_logger()

        #prepare and add blocks (including internal events)
        if blocks is not None:
            for block in blocks:
                self.add_block(block)

        #check and add connections
        if connections is not None:
            for connection in connections:
                self.add_connection(connection)

        #check and add events
        if events is not None:
            for event in events:
                self.add_event(event)

        #check if blocks from connections are in simulation
        self._check_blocks_are_managed()

        #assemble the system graph for simulation
        self._assemble_graph()


    def __str__(self):
        """String representation of the simulation using the 
        dict model format and readable json formatting
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


    def __contains__(self, other):
        """Check if blocks, connections or events are 
        already part of the simulation 

        Paramters
        ---------
        other : obj
            object to check if its part of simulation

        Returns
        -------
        bool
        """
        return (
            other in self.blocks or 
            other in self.connections or 
            other in self.events
            )


    # methods for access to metadata ----------------------------------------------

    def size(self):
        """Get size information of the simulation, such as total number 
        of blocks and dynamic states, with recursive retrieval from subsystems

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


    # logger methods --------------------------------------------------------------

    def _initialize_logger(self):
        """
        setup and configure logging
        """

        #initialize the logger
        self.logger = logging.Logger("PathSim_Simulation_Logger")

        #capture warnings from the 'warnings' module
        logging.captureWarnings(True)

        #check if logging is enabled
        if self.log:    

            #if a filename for logging is specified
            if isinstance(self.log, str):
                handler = logging.FileHandler(self.log) 
            else:
                handler = logging.StreamHandler()

            #logging format
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")   
                )

            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            self._logger_info(f"LOGGING (log: {self.log})")


    def _logger_info(self, message):
        if self.log: self.logger.info(message)


    def _logger_error(self, message, Error=None):
        if self.log: self.logger.error(message)
        if Error is not None: raise Error(message)


    def _logger_warning(self, message):
        if self.log: self.logger.warning(message)


    # visualization ---------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Plot the simulation results by calling all the blocks 
        that have visualization capabilities such as the 'Scope' 
        and 'Spectrum'.

        This is a quality of life method. Blocks can be visualized 
        individually due to the object oriented nature, but it might 
        be nice to just call the plot metho globally and look at all 
        the results at once. Also works for models loaded from an 
        external file.

        Parameters
        ----------
        args : tuple
            args for the plot methods
        kwargs : dict
            kwargs for the plot method
        """
        for block in self.blocks:
            if block: block.plot(*args, **kwargs)


    # serialization/deserialization -----------------------------------------------

    def save(self, path="", **metadata):
        """Save the dictionary representation of the simulation instance 
        to an external file
        
        Parameters
        ----------
        path : str
            filepath to save data to
        metadata : dict
            metadata for the simulation model
        """

        #add current pathsim version
        metadata["version"] = __version__

        #add current timestamp
        metadata["timestamp"] = datetime.datetime.now().isoformat()

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(**metadata), file, indent=2, ensure_ascii=False)


    @classmethod
    def load(cls, path="", **kwargs):
        """Load and instantiate a Simulation from an external file 
        in json format
        
        Parameters
        ----------
        path : str
            filepath to load data from
        kwargs : dict
            additional args for the simulation, overwriting metadata

        Returns
        -------
        out : Simulation
            reconstructed object from dict representation
        """
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file), **kwargs)
        return None


    def to_dict(self, **metadata):
        """Convert simulation to a complete model representation as a dict
        with additional metadata.

        Parameters
        ----------
        metadata : dict
            metadata for the simulation model

        Returns
        -------
        data : dict
            dict that describes the simulation model
        """
        
        #serialize system components
        blocks = [block.to_dict() for block in self.blocks]
        events = [event.to_dict() for event in self.events]
        connections = [conn.to_dict() for conn in self.connections]
                
        #create the full model
        return {
            "type": "Simulation",
            "metadata": metadata,
            "structure": {
                "blocks": blocks,
                "events": events,
                "connections": connections
                },
            "params": {
                "dt": self.dt,
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "Solver": self.Solver.__name__,
                "tolerance_fpi": self.tolerance_fpi,
                "iterations_max": self.iterations_max,
                **self.solver_kwargs
                }
            }


    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create simulation from model data dict

        Parameters
        ----------
        data : dict
            model definition in json format
        kwargs : dict
            additional args for the simulation, overwriting metadata

        Returns
        -------
        simulation : Simulation
            instance of the Simulation class with mode definition
        """
        from . import solvers

        #get system structure
        structure = data.get("structure", {})
        
        #deserialize blocks and create block ID mapping
        blocks, id_to_block = [], {}
        for block_data in structure.get("blocks", []):
            block = Block.from_dict(block_data)
            blocks.append(block)
            id_to_block[block_data["id"]] = block
        
        #deserialize connections
        connections = []
        for conn_data in structure.get("connections", []):
            
            #get source block and port
            source_block = id_to_block[conn_data["source"]["block"]]
            source_ports = conn_data["source"]["ports"]
            source = PortReference(source_block, source_ports)
            
            #get targets
            targets = []
            for trg in conn_data["targets"]:
                target_block = id_to_block[trg["block"]]
                target_ports = trg["ports"]
                targets.append(
                    PortReference(target_block, target_ports)
                    )
            
            #create connection
            connections.append(
                Connection(source, *targets)
                )
        
        #deserialize events
        events = []
        for event_data in structure.get("events", []):
            events.append(Event.from_dict(event_data))
        
        #get simulation parameters
        params = data.get("params", {})

        #get solver class
        solver_name = params.get("Solver", "SSPRK22")
        params["Solver"] = getattr(solvers, solver_name)

        #update with additional kwargs
        for name, val in kwargs.items():
            params[name] = val

        #create simulation
        return cls(
            blocks=blocks,
            connections=connections,
            events=events,
            **params
            )


    # adding system components ----------------------------------------------------

    def add_block(self, block):
        """Adds a new block to the simulation, initializes its local solver 
        instance and collects internal events of the new block. 

        This works dynamically for running simulations.

        Parameters
        ----------
        block : Block 
            block to add to the simulation
        """

        #check if block already in block list
        if block in self.blocks:
            _msg = f"block {block} already part of simulation"
            self._logger_error(_msg, ValueError)

        #initialize numerical integrator of block
        block.set_solver(self.Solver, **self.solver_kwargs)

        #add to dynamic list if solver was initialized
        if block.engine and block not in self._blocks_dyn:
            self._blocks_dyn.append(block)

        #add block to global blocklist
        self.blocks.append(block)

        #add events of block to global event list
        for event in block.get_events():
            self.add_event(event)

        #if graph already exists, it needs to be rebuilt
        if self.graph:
            self._assemble_graph()


    def add_connection(self, connection):
        """Adds a new connection to the simulaiton and checks if 
        the new connection overwrites any existing connections.

        This works dynamically for running simulations.

        Parameters
        ----------
        connection : Connection
            connection to add to the simulation
        """

        #check if connection already in connection list
        if connection in self.connections:
            _msg = f"{connection} already part of simulation"
            self._logger_error(_msg, ValueError)

        #check if connection overwrites existing connections
        for conn in self.connections:
            if connection.overwrites(conn):
                _msg = f"{connection} overwrites {conn}"
                self._logger_error(_msg, ValueError)

        #add connection to global connection list
        self.connections.append(connection)

        #if graph already exists, it needs to be rebuilt
        if self.graph:
            self._assemble_graph()


    def add_event(self, event):
        """Checks and adds a new event to the simulation.

        This works dynamically for running simulations.

        Parameters
        ----------
        event : Event
            event to add to the simulation
        """

        #check if event already in event list
        if event in self.events:
            _msg = f"{event} already part of simulation"
            self._logger_error(_msg, ValueError)

        #add event to global event list
        self.events.append(event)


    # system assembly -------------------------------------------------------------

    def _assemble_graph(self):
        """Build the internal graph representation for fast system function 
        evaluation and algebraic loop resolution.
        """

        #time the graph construction
        with Timer(verbose=False) as T:
            self.graph = Graph(self.blocks, self.connections)

        #create boosters for loop closing connections
        if self.graph.has_loops:
            self.boosters = [
                ConnectionBooster(conn) for conn in self.graph.loop_closing_connections()
            ]

        self._logger_info(
            "GRAPH (size: {}, alg. depth: {}, loop depth: {}, runtime: {})".format(
                len(self.graph), *self.graph.depth(), T
                )
            )


    # topological checks ----------------------------------------------------------

    def _check_blocks_are_managed(self):
        """Check whether the blocks that are part of the connections are 
        in the simulation block list ('self.blocks') and therefore managed 
        by the simulation.

        If not, there will be a warning in the logging.            
        """
        #collect blocks from connections
        conn_blocks = []
        for conn in self.connections:
            conn_blocks.extend(conn.get_blocks())

        #iterate set of blocks from connections (unique)
        for blk in set(conn_blocks):
            if blk not in self.blocks:
                self._logger_warning(
                    f"{blk} in 'connections' but not in 'blocks'!"
                    )


    # solver management -----------------------------------------------------------

    def _set_solver(self, Solver=None, **solver_kwargs):
        """Initialize all blocks with solver for numerical integration
        and tolerance for local truncation error ´tolerance_lte´.

        If blocks already have solvers, change the numerical integrator
        to the ´Solver´ class.

        Parameters
        ----------
        Solver : Solver
            numerical solver definition from ´pathsim.solvers´
        solver_kwargs : dict
            additional parameters for numerical solvers
        """

        #update global solver class
        if Solver is not None:
            self.Solver = Solver

        #update solver parmeters
        for k, v in solver_kwargs.items():
            self.solver_kwargs[k] = v

        #initialize dummy engine to get solver attributes
        self.engine = self.Solver()

        #iterate all blocks and set integration engines with tolerances
        self._blocks_dyn = []
        for block in self.blocks:
            block.set_solver(self.Solver, **self.solver_kwargs)
            
            #add dynamic blocks to list
            if block.engine:
                self._blocks_dyn.append(block)
        
        #logging message
        self._logger_info(
            "SOLVER (dyn. blocks: {}) -> {} (adaptive: {}, explicit: {})".format(
                len(self._blocks_dyn),
                self.engine,
                self.engine.is_adaptive, 
                self.engine.is_explicit
                )
            )


    # resetting -------------------------------------------------------------------

    def reset(self, time=0.0):
        """Reset the blocks to their initial state and the global time of 
        the simulation. 

        For recording blocks such as 'Scope', their recorded 
        data is also reset. 

        Resets linearization automatically, since resetting the blocks 
        resets their internal operators.

        Afterwards the system function is evaluated with '_update' to update
        the block inputs and outputs.

        Parameters
        ----------
        time : float
            simulation time for reset
        """

        self._logger_info(f"RESET (time: {time})")

        #reset simulation time
        self.time = time

        #reset all blocks to initial state
        for block in self.blocks:
            block.reset()

        #reset all event managers
        for event in self.events:
            event.reset()

        #evaluate system function
        self._update(self.time)


    # linearization ---------------------------------------------------------------

    def linearize(self):
        """Linearize the full system in the current simulation state 
        at the current simulation time.
        
        This is achieved by linearizing algebraic and dynamic operators 
        of the internal blocks. See definition of the 'Block' class.
    
        Before linearization, the global system function is evaluated 
        to get the blocks into the current simulation state. 
        This is only really relevant if no solving attempt has been 
        happened before.
        """
        #evaluate system function at current time
        self._update(self.time)

        #linearize all internal blocks and time it
        with Timer(verbose=False) as T:
            for block in self.blocks:
                block.linearize(self.time)

        self._logger_info(f"LINEARIZED (runtime: {T})")


    def delinearize(self):
        """Revert the linearization of the full system."""
        for block in self.blocks: 
            block.delinearize()

        self._logger_info("DELINEARIZED")


    # event system helpers --------------------------------------------------------

    def _estimate_events(self, t):
        """Estimate the time until the next.

        Parameters
        ----------
        t : float 
            evaluation time for event estimation

        Returns
        -------
        float | None
            esimated time until next event (delta)
        """

        dt_evt_min = None
        for event in self.events:

            #skip inactive events
            if not event: continue

            #get the estimate
            dt_evt = event.estimate(self.time)
            
            #check if estimate available and smaller than min
            if dt_evt_min is None or (dt_evt is not None and dt_evt < dt_evt_min):
                dt_evt_min = dt_evt

        #return time until next event or None
        return dt_evt_min


    def _buffer_events(self, t):
        """Buffer states for event monitoring before the timestep 
        is taken. 

        This is required to set reference for event monitoring and 
        backtracking for root finding.

        Parameters
        ----------
        t : float 
            evaluation time for buffering
        """

        #buffer states for event detection (with timestamp)
        for event in self.events:
            if event: event.buffer(t)


    def _detected_events(self, t):
        """Check for possible (active) events and return them chronologically, 
        sorted by their timestep ratios (closest to the initial point in time).
    
        Parameters
        ----------
        t : float
            evaluation time for event function

        Returns
        -------
        detected : list[Event]
            list of detected events within timestep
        """

        #iterate all event managers
        detected_events = []
        for event in self.events:

            #skip inactive events
            if not event: continue
            
            #check if an event is detected
            detected, close, ratio = event.detect(t)

            #event was detected during the timestep 
            if detected:
                detected_events.append([event, close, ratio])

        #return detected events sorted by ratio
        return sorted(detected_events, key=lambda e: e[-1])


    # solving system equations ----------------------------------------------------

    def _update(self, t):        
        """Distribute information within the system by evaluating the directed acyclic graph 
        (DAG) formed by the algebraic passthroughs of the blocks and resolving algebraic loops 
        through accelerated fixed-point iterations.
        
        Effectively evaluates the right hand side function of the global 
        system ODE/DAE

        .. math:: 
    
            \\begin{equnarray}
                \\dot{x} &= f(x, t) \\\\
                       0 &= g(x, t) 
            \\end{equnarray}

        by converging the whole system (´f´ and ´g´) to a fixed-point at a given point 
        in time ´t´.

        If no algebraic loops are present in the system, convergence is 
        guaranteed after the first stage (evaluation of the DAG in '_dag'). 

        Otherwise, accelerated fixed-point iterations ('_loops') are performed as a second 
        stage on the DAGs (broken cycles) of blocks that are part of or tainted by upstream 
        algebraic loops. 

        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #evaluate DAG
        self._dag(t)

        #algebraic loops -> solve them
        if self.graph.has_loops:   
            self._loops(t)


    def _dag(self, t):
        """Update the directed acyclic graph components of the system.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #perform gauss-seidel iterations without error checking
        for _, blocks_dag, connections_dag in self.graph.dag():

            #update blocks at algebraic depth (no error control)
            for block in blocks_dag:
                if block: block.update(t)

            #update connenctions at algebraic depth (data transfer)
            for connection in connections_dag:
                if connection: connection.update()


    def _loops(self, t):
        """Perform the algebraic loop solve of the system using accelerated 
        fixed-point iterations on the broken loop directed graph.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #reset accelerators of loop closing connections
        for con_booster in self.boosters:
            con_booster.reset()

        #perform solver iterations on algebraic loops
        for iteration in range(1, self.iterations_max):
            
            #iterate DAG depths of broken loops
            for depth, blocks_loop, connections_loop in self.graph.loop():

                #update blocks at algebraic depth
                for block in blocks_loop:
                    if block: block.update(t)

                #step accelerated connenctions at algebraic depth (data transfer)
                for connection in connections_loop:
                    if connection: connection.update()

            #step boosters of loop closing connections
            max_err = 0.0
            for con_booster in self.boosters:
                err = con_booster.update()
                if err > max_err:
                    max_err = err
                       
            #check convergence after first iteration
            if max_err <= self.tolerance_fpi:
                return

        #not converged -> error
        self._logger_error(
            "algebraic loop not converged (iters: {}, err: {})".format(
                self.iterations_max, max_err), 
            RuntimeError
            )


    def _solve(self, t, dt):
        """For implicit solvers, this method implements the solving step 
        of the implicit update equation.

        It already involves the evaluation of the system equation with 
        the '_update' method within the loop.

        This also tracks the evolution of the solution as an estimate 
        for the convergence via the max residual norm of the fixed point 
        equation of the previous solution.

        Parameters
        ----------
        t : float
            evaluation time for system function
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #total evaluations of system equation
        total_evals = 0

        #perform fixed-point iterations to solve implicit update equation
        for it in range(self.iterations_max):

            #evaluate system equation (this is a fixed point loop)
            self._update(t)
            total_evals += 1            

            #advance solution of implicit solver
            max_error = 0.0
            for block in self._blocks_dyn:

                #skip inactive blocks
                if not block: 
                    continue
                
                #advance solution (internal optimizer)
                error = block.solve(t, dt)
                if error > max_error:
                    max_error = error

            #check for convergence (only error)
            if max_error <= self.tolerance_fpi:
                return True, total_evals, it+1

        #not converged in 'self.iterations_max' steps
        return False, total_evals, self.iterations_max


    def steadystate(self, reset=False): 
        """Find steady state solution (DC operating point) of the system 
        by switching all blocks to steady state solver, solving the 
        fixed point equations, then switching back.

        The steady state solver forces all the temporal derivatives, i.e.
        the right hand side equation (including external inputs) of the 
        engines of dynamic blocks to zero.

        Parameters
        ----------
        reset : bool
            reset the simulation before solving for steady state (default False)
        """

        #reset the simulation before solving
        if reset:
            self.reset()

        #current solver class
        _solver = self.Solver
        
        #switch to steady state solver
        self._set_solver(SteadyState)

        #log message begin of steady state solver
        self._logger_info(f"STEADYSTATE -> STARTING (reset: {reset})")

        #solve for steady state at current time
        with Timer(verbose=False) as T:
            success, evals, iters = self._solve(self.time, self.dt)

        #catch non convergence
        if not success:
            self._logger_error(
                "STEADYSTATE -> FINISHED (success: {}, evals: {}, iters: {}, runtime: {})".format(
                    success, evals, iters, T), 
                RuntimeError
                )

        #sample result
        self._sample(self.time)

        #log message 
        self._logger_info(
            "STEADYSTATE -> FINISHED (success: {}, evals: {}, iters: {}, runtime: {})".format(
                success, evals, iters, T)
            )

        #switch back to original solver
        self._set_solver(_solver)


    # timestepping helpers --------------------------------------------------------

    def _revert(self):
        """Revert simulation state to previous timestep for adaptive solvers 
        when local truncation error is too large and timestep has to be 
        retaken with smaller timestep. 
        """

        #revert dummy engine (for history)
        self.engine.revert()

        #revert block states
        for block in self._blocks_dyn:
            if block: block.revert()


    def _sample(self, t):
        """Sample data from blocks that implement the 'sample' method such 
        as 'Scope', 'Delay' and the blocks that sample from a random 
        distribution at a given time 't'.
    
        Parameters
        ----------
        t : float
            time where to sample
        """
        for block in self.blocks:
            if block: block.sample(t)


    def _buffer_blocks(self, dt):
        """Buffer internal states of blocks before the timestep is taken. 

        This is required for runge-kutta integrators but also for the 
        zero crossing detection of the event handling system.
    
        The timesteps are also buffered because some integrators such as 
        GEAR-type methods need a history of the timesteps.

        Parameters
        ----------
        dt : float
            timestep
        """
        #buffer the dummy engine
        self.engine.buffer(dt)

        #buffer internal states of stateful blocks
        for block in self._blocks_dyn:
            if block: block.buffer(dt)


    def _step(self, t, dt):
        """Performs the 'step' method for dynamical blocks with internal 
        states that have a numerical integration engine. 

        Collects the local truncation error estimates and the timestep 
        rescale factor from the error controllers of the internal 
        intergation engines if they provide an error estimate 
        (for example embedded Runge-Kutta methods).
        
        Notes
        -----
        Not to be confused with the global 'step' method, the '_step' 
        method executes the intermediate timesteps in multistage solvers 
        such as Runge-Kutta methods.
    
        Parameters
        ----------
        t : float
            evaluation time of dynamical timestepping
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

            #skip inactive blocks
            if not block: continue

            #step the block
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


    # timestepping ----------------------------------------------------------------

    def timestep_fixed_explicit(self, dt=None):
        """Advances the simulation by one timestep 'dt' for explicit 
        fixed step solvers.

        If discrete events are detected, they are resolved immediately 
        within the timestep.

        Parameters
        ----------
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
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #default global timestep as local timestep
        if dt is None: 
            dt = self.dt

        #buffer states for event system
        self._buffer_events(self.time)

        #buffer internal states for solvers
        self._buffer_blocks(dt)

        #total function evaluations 
        total_evals = 0

        #iterate explicit solver stages with evaluation time (generator)
        for time_stage in self.engine.stages(self.time, dt):

            #evaluate system equation by fixed-point iteration
            self._update(time_stage) 
            total_evals += 1

            #timestep for dynamical blocks (with internal states)
            _1, error_norm, _3 = self._step(time_stage, dt)

        #system time after timestep
        time_dt = self.time + dt

        #evaluate system equation before sampling and event check (+dt)
        self._update(time_dt) 
        total_evals += 1

        #handle events chronologically after timestep (+dt)
        for event, _, ratio in self._detected_events(time_dt):

            #fixed timestep -> resolve event directly
            event.resolve(self.time + ratio * dt)  

            #after resolve, evaluate system equation again -> propagate event
            self._update(time_dt)  
            total_evals += 1

        #sample data after successful timestep (+dt)
        self._sample(time_dt)
 
        #increment global time and continue simulation
        self.time = time_dt 

        #max local truncation error, timestep rescale, successful step
        return True, error_norm, 1.0, total_evals, 0


    def timestep_fixed_implicit(self, dt=None): 
        """Advances the simulation by one timestep 'dt' for implicit 
        fixed step solvers.

        If discrete events are detected, they are resolved immediately 
        within the timestep.

        Parameters
        ----------
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
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #successful by default
        success = True

        #default global timestep as local timestep
        if dt is None: 
            dt = self.dt

        #buffer states for event system
        self._buffer_events(self.time)

        #buffer internal states for solvers
        self._buffer_blocks(dt)

        #total function evaluations and implicit solver iterations
        total_evals, total_solver_its = 0, 0

        #iterate explicit solver stages with evaluation time (generator)
        for time_stage in self.engine.stages(self.time, dt):

            #solve implicit update equation and get iteration count
            success, evals, solver_its = self._solve(time_stage, dt)

            #warning if implicit solver didnt converge in timestep
            if not success:
                self._logger_warning(
                    f"implicit solver not converged in {solver_its} iterations!"
                    )

            #count solver iterations and function evaluations
            total_solver_its += solver_its
            total_evals += evals

            #timestep for dynamical blocks (with internal states)
            _1, error_norm, _3 = self._step(time_stage, dt)

        #system time after timestep
        time_dt = self.time + dt

        #evaluate system equation before sampling and event check (+dt)
        self._update(time_dt) 
        total_evals += 1

        #handle events chronologically after timestep (+dt)
        for event, _, ratio in self._detected_events(time_dt):

            #fixed timestep -> resolve event directly
            event.resolve(self.time + ratio * dt)  

            #after resolve, evaluate system equation again -> propagate event
            self._update(time_dt)  
            total_evals += 1    

        #sample data after successful timestep (+dt)
        self._sample(time_dt)
 
        #increment global time and continue simulation
        self.time = time_dt 

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, 1.0, total_evals, total_solver_its


    def timestep_adaptive_explicit(self, dt=None): 
        """Advances the simulation by one timestep 'dt' for explicit 
        adaptive solvers.

        If the local truncation error of the solver exceeds the tolerances 
        set in the 'solver_kwargs', the simulation state is reverted to the 
        state that was buffered (`_buffer(time, dt)`) at the beginning of 
        the timestep.

        If discrete events are detected, the chronologically first event is 
        handled only. The event location (in time) is approached adaptively 
        by reverting the step and adjusting the stepsize (this is equivalent 
        to the secant method for finding zeros of the event function) until 
        the tolerance of the event is satisfied (close==True).

        Parameters
        ----------
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
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #default global timestep as local timestep
        if dt is None: 
            dt = self.dt

        #buffer states for event system
        self._buffer_events(self.time)

        #buffer internal states for solvers
        self._buffer_blocks(dt)

        #total function evaluations and implicit solver iterations
        total_evals = 0

        #iterate explicit solver stages with evaluation time (generator)
        for time_stage in self.engine.stages(self.time, dt):

            #evaluate system equation by fixed-point iteration
            self._update(time_stage) 
            total_evals += 1

            #timestep for dynamical blocks (with internal states)
            success, error_norm, scale = self._step(time_stage, dt)

        #if step not successful -> roll back timestep
        if not success:
            self._revert()
            self._update(self.time) 
            total_evals += 1
            return False, error_norm, scale, total_evals, 0

        #system time after timestep
        time_dt = self.time + dt

        #evaluate system equation before sampling and event check (+dt)
        self._update(time_dt) 
        total_evals += 1

        #handle detected events chronologically after timestep (+dt)
        for event, close, ratio in self._detected_events(time_dt):

            #close enough to event (ratio approx 1.0) -> resolve it
            if close:
                event.resolve(time_dt)

                #after resolve, evaluate system equation again -> propagate event
                self._update(time_dt) 
                total_evals += 1
    
            #not close enough -> roll back timestep (secant step)
            else:
                self._revert()
                self._update(self.time) 
                total_evals += 1
                return False, error_norm, ratio, total_evals, 0
        
        #sample data after successful timestep (+dt)
        self._sample(time_dt)

        #increment global time and continue simulation
        self.time = time_dt    

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, scale, total_evals, 0


    def timestep_adaptive_implicit(self, dt=None): 
        """Advances the simulation by one timestep 'dt' for implicit 
        adaptive solvers.

        If the local truncation error of the solver exceeds the tolerances 
        set in the 'solver_kwargs', the simulation state is reverted to the 
        state that was buffered (`_buffer(time, dt)`) at the beginning of 
        the timestep.

        If the solution of the implicit update equation in 'solve' doesnt 
        converge, the timestep is also considered unsuccessful. Then it is 
        reverted and the timestep is halfed.

        If discrete events are detected, the chronologically first event is 
        handled only. The event location (in time) is approached adaptively 
        by reverting the step and adjusting the stepsize (this is equivalent 
        to the secant method for finding zeros of the event function) until 
        the tolerance of the event is satisfied (close==True).

        Parameters
        ----------
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
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations
        """

        #default global timestep as local timestep
        if dt is None: 
            dt = self.dt

        #buffer states for event system
        self._buffer_events(self.time)

        #buffer internal states for solvers
        self._buffer_blocks(dt)

        #total function evaluations and implicit solver iterations
        total_evals, total_solver_its = 0, 0

        #iterate explicit solver stages with evaluation time (generator)
        for time_stage in self.engine.stages(self.time, dt):

            #solve implicit update equation and get iteration count
            success, evals, solver_its = self._solve(time_stage, dt)

            #count solver iterations and function evaluations
            total_solver_its += solver_its
            total_evals += evals

            #if solver did not converge -> quit early (adaptive only)
            if not success:
                self._revert()
                self._update(self.time) 
                return False, 0.0, 0.5, total_evals+1, total_solver_its  

            #timestep for dynamical blocks (with internal states)
            success, error_norm, scale = self._step(time_stage, dt)

        #if step not successful -> roll back timestep
        if not success:
            self._revert()
            self._update(self.time) 
            return False, error_norm, scale, total_evals+1, total_solver_its

        #system time after timestep
        time_dt = self.time + dt

        #evaluate system equation before sampling and event check (+dt)
        self._update(time_dt) 
        total_evals += 1

        #handle detected events chronologically after timestep (+dt)
        for event, close, ratio in self._detected_events(time_dt):

            #close enough to event (ratio approx 1) -> resolve it
            if close:
                event.resolve(time_dt)

                #after resolve, evaluate system equation again -> propagate event
                self._update(time_dt) 
                total_evals += 1
    
            #not close enough -> roll back timestep (secant step)
            else:
                self._revert()
                self._update(self.time) 
                total_evals += 1
                return False, error_norm, ratio, total_evals, total_solver_its

        #sample data after successful timestep (+dt)
        self._sample(time_dt)

        #increment global time and continue simulation
        self.time = time_dt    

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, scale, total_evals, total_solver_its


    def timestep(self, dt=None, adaptive=True):
        """Advances the transient simulation by one timestep 'dt'. 
        
        Automatic stepping method selection based on 
        selected `Solver`.

        Parameters
        ----------
        dt : float
            timestep size for transient simulation
        adaptive : bool
            explicitly select the addaptive timestepping branch

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        total_evals : int
            total number of system evaluations
        total_solver_its : int
            total number of implicit solver iterations            
        """
        if adaptive and self.engine.is_adaptive:
            if self.engine.is_explicit:
                return self.timestep_adaptive_explicit(dt)
            else:                
                return self.timestep_adaptive_implicit(dt)
        else:
            if self.engine.is_explicit:
                return self.timestep_fixed_explicit(dt)
            else:                
                return self.timestep_fixed_implicit(dt)


    def step(self, dt=None, adaptive=True):
        """Wraps 'Simulation.timestep' for backward compatibility"""
        self._logger_warning(
            "'Simulation.step' method will be deprecated in next release, use 'Simulation.timestep' instead!"
            )
        return self.timestep(dt, adaptive)


    # simulation execution --------------------------------------------------------

    def run(self, duration=10, reset=False, adaptive=True):
        """Perform multiple simulation timesteps for a given 'duration'.
        
        Tracks the total number of block evaluations (proxy for function 
        calls, although larger, since one function call of the system equation 
        consists of many block evaluations) and the total number of solver
        iterations for implicit solvers.

        Additionally the progress of the simulation is tracked by a custom
        'ProgressTracker' class that is a dynamic generator and interfaces 
        the logging system.

        Parameters
        ----------
        duration : float
            simulation time (in time units)
        reset : bool
            reset the simulation before running (default False)
        adaptive : bool
            use adaptive timesteps if solver is adaptive (default True)

        Returns
        -------
        stats : dict
            stats of simulation run tracked by the ´ProgressTracker´ 
        """

        #reset the simulation before running it
        if reset:
            self.reset()

        #make an adaptive run?
        _adaptive = adaptive and self.engine.is_adaptive

        #simulation start and end time
        start_time, end_time = self.time, self.time + duration

        #effective timestep for duration
        _dt = self.dt

        #initial system function evaluation 
        self._update(self.time)
        initial_evals = 1

        #catch and resolve initial events
        for event, *_ in self._detected_events(self.time):

            #resolve events directly
            event.resolve(self.time)

            #evaluate system function again -> propagate event
            self._update(self.time) 
            initial_evals += 1
    
        #sampling states and inputs at 'self.time == starting_time' 
        self._sample(self.time)

        #initialize progress tracker
        tracker = ProgressTracker(
            total_duration=duration, 
            description="TRANSIENT", 
            logger=self.logger,
            log=self.log
            )

        #enter tracker context
        with tracker:

            #iterate progress tracker generator until 'progress >= 1.0' is reached
            for _ in tracker:

                #advance the simulation by one (effective) timestep '_dt'
                success, error_norm, scale, *_ = self.timestep(
                    dt=_dt, 
                    adaptive=_adaptive
                    )

                #perform adaptive rescale
                if _adaptive:            

                    #if no error estimate and rescale -> back to default timestep
                    if not error_norm and scale == 1:
                        _dt = self.dt

                    #rescale due to error control
                    _dt = scale * _dt

                    #estimate time until next event and adjust timestep
                    _dt_evt = self._estimate_events(self.time)
                    if _dt_evt is not None and _dt_evt < _dt:
                        _dt = _dt_evt
                        
                    #rescale if in danger of overshooting 'end_time' at next step
                    if self.time + _dt > end_time:
                        _dt = end_time - self.time

                    #apply bounds to timestep after rescale
                    _dt = np.clip(_dt, self.dt_min, self.dt_max)

                #compute simulation progress
                progress = np.clip((self.time - start_time)/duration, 0.0, 1.0)

                #update the tracker
                tracker.update(progress, success=success)

        return tracker.stats