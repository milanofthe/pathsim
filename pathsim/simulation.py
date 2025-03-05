#########################################################################################
##
##                     MAIN SIMULATION ENGINE FOR TRANSIENT ANALYSIS
##                                   (simulation.py)
##
##                 This module contains the simulation class that handles
##                the blocks and connections and the timestepping methods.
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

import json
import datetime
import logging

from .utils.utils import path_length_dfs
from .utils.debugging import Timer
from .utils.progresstracker import ProgressTracker

from .solvers import SSPRK22, SteadyState

from .blocks._block import Block
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

    Parameters
    ----------
    blocks : list[Block] 
        blocks that make up the system
    connections : list[Connection] 
        connections that connect the blocks
    events : list[Event]
        list of event trackers (zero crossing detection)
    dt : float
        transient simulation timestep in time units
    dt_min : float
        lower bound for timestep, default '0.0'
    dt_max : float
        upper bound for timestep, default 'None'
    Solver : Solver 
        solver for numerical integration from pathsim.solvers
    tolerance_fpi : float
        absolute tolerance for convergence of fixed-point iterations
    iterations_min : int
        minimum number of fixed-point iterations for system function evaluation
    iterations_max : int
        maximum allowed number of fixed-point iterations for system function evaluation
    log : bool, string
        flag to enable logging (alternatively a path can be specified)
    solver_args : dict
        additional parameters for numerical solvers such as abs and rel tolerance

    Attributes
    ----------
    time : float
        global simulation time
    path_length : int
        estimated length of the longest algebraic path
    engine : Solver
        global integrator instance
    logger : logging.Logger
        global simulation logger

    """

    def __init__(self, 
                 blocks=None, 
                 connections=None, 
                 events=None,
                 dt=0.01, 
                 dt_min=1e-16, 
                 dt_max=None, 
                 Solver=SSPRK22, 
                 tolerance_fpi=1e-12, 
                 iterations_min=None, 
                 iterations_max=200, 
                 log=True,
                 **solver_args
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

        #numerical integrator instance -> initialized later
        self.engine = None

        #error tolerance for fixed point loop and implicit solver
        self.tolerance_fpi = tolerance_fpi

        #additional solver parameters
        self.solver_args = solver_args

        #iterations for fixed-point loop
        self.iterations_min = iterations_min
        self.iterations_max = iterations_max

        #enable logging flag
        self.log = log

        #initial simulation time
        self.time = 0.0

        #length of the longest algebraic path
        self.path_length = 1

        #initialize logging for logging mode
        self._initialize_logger()

        #prepare and add blocks (including internal events)
        if blocks is not None:
            for block in blocks:
                self.add_block(
                    block, 
                    recompute_path=False
                    )

        #check and add connections
        if connections is not None:
            for connection in connections:
                self.add_connection(
                    connection, 
                    recompute_path=False
                    )

        #check and add events
        if events is not None:
            for event in events:
                self.add_event(event)

        #set numerical integration solver
        self._set_solver()

        #find length of longest algebraic path
        self._algebraic_path_length()


    def __str__(self):
        """String representation of the simulation using the 
        dict model format and readable json formatting
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


    # logger methods --------------------------------------------------------------

    def _initialize_logger(self):
        """
        setup and configure logging
        """

        #initialize the logger
        self.logger = logging.Logger("PathSim_Simulation_Logger")

        #check if logging is selected
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

            self._logger_info("LOGGING enabled")


    def _logger_info(self, message):
        if self.log: self.logger.info(message)


    def _logger_error(self, message, Error=None):
        if self.log: self.logger.error(message)
        if Error is not None: raise Error(message)


    def _logger_warning(self, message):
        if self.log: self.logger.warning(message)


    # serialization/deserialization -----------------------------------------------

    def save(self, path=""):
        """Save the dictionary representation of the simulation instance 
        to an external file
        
        Parameters
        ----------
        path : str
            filepath to save data to
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


    @classmethod
    def load(cls, path=""):
        """Load and instantiate a Simulation from an external file 
        in json format
        
        Parameters
        ----------
        path : str
            filepath to load data from

        Returns
        -------
        out : Simulation
            reconstructed object from dict representation
        """
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
        return None


    def to_dict(self, name="Model", description=""):
        """Convert simulation to a complete model representation as a dict

        Parameters
        ----------
        name : str
            model name
        description : str
            description of the model

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
        data = {
            "metadata": {
                "name": name,
                "description": description,
                "created": datetime.datetime.now().isoformat()
            },
            "blocks": blocks,
            "connections": connections,
            "events": events,
            "simulation": {
                "dt": self.dt,
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "solver": self.Solver.__name__,
                "tolerance_fpi": self.tolerance_fpi,
                "iterations_min": self.iterations_min,
                "iterations_max": self.iterations_max
            }
        }
        
        return data


    @classmethod
    def from_dict(cls, data):
        """Create simulation from model data dict

        Parameters
        ----------
        data : dict
            model definition in json format

        Returns
        -------
        simulation : Simulation
            instance of the Simulation class with mode definition
        """
        from . import solvers
        
        #deserialize blocks and create block ID mapping
        blocks, id_to_block = [], {}
        for block_data in data["blocks"]:
            block = Block.from_dict(block_data)
            blocks.append(block)
            id_to_block[block_data["id"]] = block
        
        #deserialize connections
        connections = []
        for conn_data in data["connections"]:
            
            #get source block and port
            source_block = id_to_block[conn_data["source"]["block"]]
            source_port = conn_data["source"]["port"]
            
            #get targets
            targets = []
            for trg in conn_data["targets"]:
                target_block = id_to_block[trg["block"]]
                target_port = trg["port"]
                targets.append((target_block, target_port))
            
            #create connection
            connections.append(Connection((source_block, source_port), *targets))
        
        #deserialize events
        events = []
        for event_data in data.get("events", []):
            events.append(Event.from_dict(event_data))
        
        #get simulation parameters
        sim_data = data.get("simulation", {})
        
        #get solver class
        solver_name = sim_data.get("solver", "SSPRK22")
        Solver = getattr(solvers, solver_name)
        
        #create simulation
        return cls(
            blocks=blocks,
            connections=connections,
            events=events,
            dt=sim_data.get("dt", 0.01),
            dt_min=sim_data.get("dt_min", 0.0),
            dt_max=sim_data.get("dt_max", None),
            Solver=Solver,
            tolerance_fpi=sim_data.get("tolerance_fpi", 1e-12),
            iterations_min=sim_data.get("iterations_min", None),
            iterations_max=sim_data.get("iterations_max", 200)
            )


    # adding system components ----------------------------------------------------

    def add_block(self, block, recompute_path=True):
        """Adds a new block to the simulation, initializes its local solver 
        instance and collects internal events of the new block. 

        This works dynamically for running simulations.

        Recomputes the length of the longest internal algebraic signal path
        if specified in the argument. This is for dynamically adding blocks
        mid simulation.

        Parameters
        ----------
        block : Block 
            block to add to the simulation
        recompute_path : bool 
            flag for recomputing the algebraic path length
        """

        #check if block already in block list
        if block in self.blocks:
            _msg = f"block {block} already part of simulation"
            self._logger_error(_msg, ValueError)

        #initialize numerical integrator of block
        block.set_solver(self.Solver, **self.solver_args)

        #add block to global blocklist
        self.blocks.append(block)

        #add events of block to global event list
        for event in block.get_events():
            self.add_event(event)

        #recompute algebraic path length
        if recompute_path:
            self._algebraic_path_length()


    def add_connection(self, connection, recompute_path=True):
        """Adds a new connection to the simulaiton and checks if 
        the new connection overwrites any existing connections.

        This works dynamically for running simulations.

        Recomputes the length of the longest internal algebraic 
        signal path if specified in the argument. This is for 
        dynamically adding connections mid simulation.

        Parameters
        ----------
        connection : Connection
            connection to add to the simulation
        recompute_path : bool 
            flag for recomputing the algebraic path length
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

        #recompute algebraic path length
        if recompute_path:
            self._algebraic_path_length()


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


    # topological checks ----------------------------------------------------------

    def _algebraic_path_length(self):
        """Perform recursive depth first search to compute the length of the 
        longest signal path through algebraic (instant time) blocks, 
        information can travel within a single timestep.
    
        The depth first search leverates the '__len__' method of the blocks 
        for contribution of each block to the total signal path. 

        This enables 'Subsystem' blocks to recursively propagate their internal 
        length upward the hierarchies.

        The result 'max_path_length' can be used as a an estimate for the 
        minimum number of fixed-point iterations in the '_update' method in 
        the main simulation loop.
        """

        #iterate all possible starting blocks (nodes of directed graph)
        for block in self.blocks:

            #recursively compute the longest path via depth first search
            _path_length = path_length_dfs(self.connections, block)

            #update global algebraic path length
            if _path_length > self.path_length:
                self.path_length = _path_length
        
        #logging message
        self._logger_info(f"ALGEBRAIC PATH LENGTH {self.path_length}")
        
        #set 'iterations_min' for fixed-point loop if not provided globally
        if self.iterations_min is None:
            self.iterations_min = self.path_length


    # solver management -----------------------------------------------------------

    def _set_solver(self, Solver=None, **solver_args):
        """Initialize all blocks with solver for numerical integration
        and tolerance for local truncation error ´tolerance_lte´.

        If blocks already have solvers, change the numerical integrator
        to the ´Solver´ class.

        Parameters
        ----------
        Solver : Solver
            numerical solver definition from ´pathsim.solvers´
        solver_args : dict
            additional parameters for numerical solvers
        """

        #update global solver class
        if Solver is not None:
            self.Solver = Solver

        #update solver parmeters
        for k, v in solver_args.items():
            self.solver_args[k] = v

        #initialize dummy engine to get solver attributes
        self.engine = self.Solver()

        #iterate all blocks and set integration engines with tolerances
        for block in self.blocks:
            block.set_solver(self.Solver, **self.solver_args)

        #logging message
        self._logger_info(
            "SOLVER -> {}, adaptive={}, implicit={}".format(
                self.engine,
                self.engine.is_adaptive, 
                not self.engine.is_explicit
                )
            )


    # resetting -------------------------------------------------------------------

    def reset(self):
        """Reset the blocks to their initial state and the global time of 
        the simulation. 

        For recording blocks such as 'Scope', their recorded 
        data is also reset. 

        Afterwards the system function os evaluated with '_update' to update
        the block inputs and outputs.
        """

        self._logger_info("RESET, time -> 0.0")

        #reset simulation time
        self.time = 0.0

        #reset all blocks to initial state
        for block in self.blocks:
            block.reset()

        #reset all event managers
        for event in self.events:
            event.reset()

        #evaluate system function
        self._update(0.0)


    # event system ----------------------------------------------------------------

    def _events(self, t):
        """Check for possible (active) events and return them chronologically, 
        sorted by their timestep ratios (closest to the initial point in time).
    
        Parameters
        ----------
        t : float
            evaluation time for event function
        """

        #iterate all event managers
        detected_events = []
        for event in self.events:

            #skip inactive events
            if not event:
                continue
            
            #check if an event is detected
            detected, close, ratio = event.detect(t)

            #event was detected during the timestep 
            if detected:
                detected_events.append([event, close, ratio])

        #return detected events sorted by ratio
        return sorted(detected_events, key=lambda e: e[-1])


    # solving system equations ----------------------------------------------------

    def _update(self, t):
        """Fixed-point iterations to resolve algebraic loops and distribute 
        information within the system.
        
        Effectively evaluates the right hand side function of the global 
        system ODE/DAE

            dx/dt = f(x, t) <- this one (ODE system function)
                0 = g(x, t) <- and this one (algebraic constraints)

        by converging the whole system to a fixed-point at a given point 
        in time 't'.

        If no algebraic loops are present in the system, it usually converges
        already after 'iterations_min' as long as the path length has been 
        used as an estimate for the minimum number of iterations.

        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #perform minimum number of fixed-point iterations without error checking
        for _iteration in range(self.iterations_min):
                        
            #update connenctions (data transfer)
            for connection in self.connections:
                if connection: connection.update()

            #update all blocks
            for block in self.blocks:
                if block: block.update(t)

        #perform fixed-point iterations until convergence with error checking
        for iteration in range(self.iterations_min, self.iterations_max):
                        
            #update connenctions (data transfer)
            for connection in self.connections:
                if connection: connection.update()

            #update instant time blocks
            max_error = 0.0
            for block in self.blocks:
                if block:
                    error = block.update(t)
                    if error > max_error:
                        max_error = error

            #return number of iterations if converged
            if max_error <= self.tolerance_fpi:
                return iteration+1

        #not converged
        self._logger_error(
            "fixed-point loop in '_update' not converged, iters={}, err={}".format(
                iteration+1, 
                max_error
                ), 
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
        for iteration in range(self.iterations_max):

            #evaluate system equation (this is a fixed point loop)
            total_evals += self._update(t)

            #advance solution of implicit solver
            max_error = 0.0
            for block in self.blocks:
                if block:
                    error = block.solve(t, dt)
                    if error > max_error:
                        max_error = error

            #check for convergence (only error)
            if max_error <= self.tolerance_fpi:
                return True, total_evals, iteration + 1

        #not converged in 'self.iterations_max' steps
        return False, total_evals, iteration + 1


    def steadystate(self, reset=True): 
        """Find steady state solution (DC operating point) of the system 
        by switching all blocks to steady state solver, solving the 
        fixed point equations, then switching back.

        The steady state solver forces all the temporal derivatives, i.e.
        the right hand side equation (including external inputs) of the 
        engines of dynamic blocks to zero.

        Parameters
        ----------
        reset : bool
            reset the simulation before solving for steady state
        """
    
        #reset the simulation before solving
        if reset:
            self.reset()

        #current solver class
        _solver = self.Solver
        
        #switch to steady state solver
        self._set_solver(SteadyState)    

        #log message begin of steady state solver
        self._logger_info(f"STEADYSTATE start, reset={reset}")

        #solve for steady state at current time
        with Timer(verbose=False) as T:
            success, evals, iters = self._solve(self.time, self.dt)

        #catch non convergence
        if not success:
            self._logger_error(
                "STEADYSTATE not converged, evals={}, iters={}, runtime={}".format(
                    evals, 
                    iters, 
                    T.readout
                    ), 
                RuntimeError
                )

        #sample result
        self._sample(self.time)

        #log message 
        self._logger_info(
            "STEADYSTATE success, evals={}, iters={}, runtime={}".format(
                evals, 
                iters, 
                T.readout
                )
            )

        #switch back to original solver
        self._set_solver(_solver)


    # timestepping ----------------------------------------------------------------

    def _revert(self):
        """Revert simulation state to previous timestep for adaptive solvers 
        when local truncation error is too large and timestep has to be 
        retaken with smaller timestep.
        """
        for block in self.blocks:
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


    def _buffer(self, t, dt):
        """Buffer internal states of blocks and buffer states for event 
        monitoring before the timestep is taken. 

        This is required for runge-kutta integrators but also for the 
        zero crossing detection of the event handling system.
    
        The timesteps are also buffered because some integrators such as 
        GEAR-type methods need a history of the timesteps.

        Parameters
        ----------
        t : float 
            evaluation time for buffering
        dt : float
            timestep
        """

        #buffer internal states of stateful blocks
        for block in self.blocks:
            if block: block.buffer(dt)

        #buffer states for event detection (with timestamp)
        for event in self.events:
            if event: event.buffer(t)


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
        for block in self.blocks:

            #skip inactive blocks
            if not block:
                continue

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


    def step_fixed(self, dt=None):
        """Advances the simulation by one timestep 'dt' for fixed step solvers.

        Selects between implicit and explicit solvers. Implicit solvers have 
        an additional loop for solving the implicit update equation in each 
        timestep.

        If discrete events are detected, they are resolved immediately within 
        the timestep.

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

        #buffer internal states for solvers and event system
        self._buffer(self.time, dt)

        #total function evaluations and implicit solver iterations
        total_evals, total_solver_its = 0, 0

        #iterate explicit solver stages with evaluation time (generator)
        for time in self.engine.stages(self.time, dt):

            #explicit solver stepping loop
            if self.engine.is_explicit:

                #evaluate system equation by fixed-point iteration
                total_evals += self._update(time) 

            #implicit solver stepping loop
            else:

                #solve implicit update equation and get iteration count
                success, evals, solver_its = self._solve(time, dt)

                #count solver iterations and function evaluations
                total_solver_its += solver_its
                total_evals += evals

            #timestep for dynamical blocks (with internal states)
            success, error_norm, scale = self._step(time, dt)

        #evaluate system equation before sampling and event check (+dt)
        total_evals += self._update(self.time + dt) 

        #handle events chronologically after timestep (+dt)
        for event, _, ratio in self._events(self.time + dt):

            #fixed timestep -> resolve event directly
            event.resolve(self.time + ratio * dt)  

            #after resolve, evaluate system equation again -> propagate event
            total_evals += self._update(self.time + dt)      

        #sample data after successful timestep (+dt)
        self._sample(self.time + dt)
 
        #increment global time and continue simulation
        self.time += dt 

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, scale, total_evals, total_solver_its


    def step_adaptive(self, dt=None):
        """Advances the simulation by one timestep 'dt' for adaptive solvers.

        Selects between implicit and explicit solvers. Implicit solvers have an 
        additional loop for solving the implicit update equation in each timestep.
    
        If the local truncation error of the solver exceeds the tolerances
        set in the 'solver_args', simulation state is reverted to the state 
        before the 'step' method was called. 

        If the solver is implicit and the solution of the implicit update 
        equation in 'solve' doesnt converge, the timestep is also considered 
        unsuccessful. Then it is reverted and the timestep is halfed.

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

        #buffer internal states for solvers and event system
        self._buffer(self.time, dt)

        #total function evaluations and implicit solver iterations
        total_evals, total_solver_its = 0, 0

        #iterate explicit solver stages with evaluation time (generator)
        for time in self.engine.stages(self.time, dt):

            #explicit solver stepping loop
            if self.engine.is_explicit:

                #evaluate system equation by fixed-point iteration
                total_evals += self._update(time) 

            #implicit solver stepping loop
            else:

                #solve implicit update equation and get iteration count
                success, evals, solver_its = self._solve(time, dt)

                #count solver iterations and function evaluations
                total_solver_its += solver_its
                total_evals += evals

                #if solver did not converge -> quit early (adaptive only)
                if not success:
                    error_norm, scale = 0.0, 0.5
                    break    

            #timestep for dynamical blocks (with internal states)
            success, error_norm, scale = self._step(time, dt)

        #if step not successful and adaptive -> roll back timestep
        if not success:
            self._revert()
            return False, error_norm, scale, total_evals, total_solver_its

        #evaluate system equation before sampling and event check (+dt)
        total_evals += self._update(self.time + dt) 

        #handle events chronologically after timestep (+dt)
        for event, close, ratio in self._events(self.time + dt):

            #close enough to event -> resolve it
            if close:
                event.resolve(self.time + ratio * dt)

                #after resolve, evaluate system equation again -> propagate event
                total_evals += self._update(self.time + dt) 
    
            #not close enough -> roll back timestep (secant step)
            else:
                self._revert()
                scale = min(scale, ratio)
                return False, error_norm, scale, total_evals, total_solver_its
        
        #sample data after successful timestep (+dt)
        self._sample(self.time + dt)

        #increment global time and continue simulation
        self.time += dt    

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, scale, total_evals, total_solver_its


    def step(self, dt=None, adaptive=False):
        """Advances the simulation by one timestep 'dt'. 
        
        Wraps the 'step_fixed' and 'step_adaptive' methods
        and can be called from the outside in case the simulation
        should be advanced one step at a time.
        
        Parameters
        ----------
        dt : float
            timestep
        adaptive : bool
            flag for adaptive timestepping

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
        if adaptive: return self.step_adaptive(dt)
        else: return self.step_fixed(dt)


    def run(self, duration=10, reset=True, adaptive=True):
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
            reset the simulation before running
        adaptive : bool
            use adaptive timesteps if solver is adaptive

        Returns
        -------
        stats : dict
            stats of simulation run tracked by the ´ProgressTracker´ 
        """

        #reset the simulation before running it
        if reset:
            self.reset()

        #select simulation stepping method
        adaptive = adaptive and self.engine.is_adaptive

        #log message solver selection
        self._logger_info(f"TRANSIENT duration={duration}")

        #simulation start and end time
        start_time, end_time = self.time, self.time + duration

        #effective timestep for duration
        _dt = self.dt

        #initial system function evaluation 
        initial_evals = self._update(self.time)

        #catch and resolve initial events
        for event, *_ in self._events(self.time):

            #resolve events directly
            event.resolve(self.time)

            #evaluate system function again -> propagate event
            initial_evals += self._update(self.time) 
    
        #sampling states and inputs at 'self.time == starting_time' 
        self._sample(self.time)

        #initialize progress tracker
        tracker = ProgressTracker(
            logger=self.logger, 
            log_interval=10, 
            function_evaluations=initial_evals
            )

        #iterate progress tracker generator until 'progress >= 1.0' is reached
        for _ in tracker:

            #rescale effective timestep if in danger of overshooting 'end_time'
            if self.time + _dt > end_time:
                _dt = end_time - self.time

            #perform adaptive timestep including rescale
            if adaptive:

                #advance the simulation by one (effective) timestep '_dt'
                success, _, scale, evals, solver_its = self.step_adaptive(_dt)

                #apply bounds to timestep after rescale
                _dt = np.clip(scale*_dt, self.dt_min, self.dt_max)

            #perform fixed timestep
            else:

                #advance the simulation by one (effective) timestep '_dt'
                success, _, scale, evals, solver_its = self.step_fixed(_dt)

            #calculate progress and update progress tracker
            tracker.check(
                progress=(self.time - start_time)/duration, 
                success=success, 
                function_evaluations=evals, 
                solver_iterations=solver_its
                )

        return tracker.stats