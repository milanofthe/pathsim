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
import logging

from .utils.utils import path_length_dfs
from .utils.progresstracker import ProgressTracker
from .solvers import SSPRK22


# TRANSIENT SIMULATION CLASS ============================================================

class Simulation:
    """
    Class that performs transient analysis of the dynamical system, defined by the 
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

    INPUTS:
        blocks         : (list of 'Block' objects) blocks that make up the system
        connections    : (list of 'Connection' objects) connections that connect the blocks
        dt             : (float) transient simulation timestep in time units
        dt_min         : (float) lower bound for timestep, default '0.0'
        dt_max         : (float) upper bound for timestep, default 'None'
        Solver         : ('Solver' class) solver for numerical integration from pathsim.solvers
        tolerance_fpi  : (float) absolute tolerance for convergence of fixed-point iterations
        iterations_min : (int) minimum number of fixed-point iterations for system function evaluation
        iterations_max : (int) maximum allowed number of fixed-point iterations for system function evaluation
        log            : (bool, string) flag to enable logging (alternatively a path can be specified)
        solver_args    : (dict) additional parameters for numerical solvers such as abs and rel tolerance
    """

    def __init__(self, 
                 blocks=None, 
                 connections=None, 
                 dt=0.01, 
                 dt_min=0.0, 
                 dt_max=None, 
                 Solver=SSPRK22, 
                 tolerance_fpi=1e-12, 
                 iterations_min=None, 
                 iterations_max=200, 
                 log=True,
                 **solver_args
                 ):

        #system definition
        self.blocks = [] if blocks is None else blocks
        self.connections = [] if connections is None else connections

        #simulation timestep and bounds
        self.dt = dt
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

        #setup everything 
        self._setup()


    def __str__(self):
        return "\n".join([str(block) for block in self.blocks])


    # simulation setup ------------------------------------------------------------

    def _setup(self):
        """
        Initialize the logger, check the connections for validity, initialize 
        the numerical integrators within the dynamical blocks and compute the 
        internal path length of the system.

        This is very lightweight.
        """
 
        #initialize logging for logging mode
        self._initialize_logger()

        #check if connections are valid
        self._check_connections()

        #set numerical integration solver to all blocks 
        self._set_solver()

        #compute the length of the longest path in the system
        self._estimate_path_length()


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
            filename = self.log if isinstance(self.log, str) else None
            handler = logging.FileHandler(filename) if filename else logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)

            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            self._logger_info("LOGGING enabled")


    def _logger_info(self, message):
        if self.log: self.logger.info(message)


    def _logger_error(self, message):
        if self.log: self.logger.error(message)


    def _logger_warning(self, message):
        if self.log: self.logger.warning(message)


    # adding blocks and connections -----------------------------------------------

    def add_block(self, block):
        """
        Adds a new block to an existing 'Simulation' instance and initializes the solver.

        INPUTS:
            block : ('Block' instance) block to add to the simulation
        """

        #check if block already in block list
        if block in self.blocks:
            _msg = f"block {block} already part of simulation"
            self._logger_error(_msg)
            raise ValueError(_msg)

        #initialize numerical integrator of block
        block.set_solver(self.Solver, **self.solver_args)

        #add block to global blocklist
        self.blocks.append(block)


    def add_connection(self, connection):
        """
        Adds a new connection to an existing 'Simulation' instance.

        INPUTS:
            connection : ('Connection' instance) connection to add to the simulation
        """

        #check if connection already in block list
        if connection in self.connections:
            _msg = f"{connection} already part of simulation"
            self._logger_error(_msg)
            raise ValueError(_msg)

        #check if connection overwrites existing connections
        for conn in self.connections:
            if connection.overwrites(conn):
                _msg = f"{connection} overwrites {conn}"
                self._logger_error(_msg)
                raise ValueError(_msg)

        #add connection to global connection list
        self.connections.append(connection)


    # topological checks ----------------------------------------------------------

    def _check_connections(self):
        """
        Check if connections are valid and if there is no input port that recieves 
        multiple outputs and could be overwritten unintentionally.

        If multiple outputs are assigned to the same input, a 'ValueError' is raised.
        """

        #iterate connections and check if they are valid
        for i, conn_1 in enumerate(self.connections):

            #check if connections overwrite each other and raise exception
            for conn_2 in self.connections[(i+1):]:
                if conn_1.overwrites(conn_2):
                    _msg = f"{conn_1} overwrites {conn_2}"
                    self._logger_error(_msg)
                    raise ValueError(_msg)


    def _estimate_path_length(self):
        """
        Perform recursive depth first search to compute the length of the 
        longest signal path over instant time blocks, information can travel 
        within a single timestep.
    
        The depth first search leverates the '__len__' method of the blocks 
        for contribution of each block to the total signal path. 
        This enables 'Subsystem' blocks to propagate their internal length upward.

        The result 'max_path_length' can be used as a an estimate for the 
        minimum number of fixed-point iterations in the '_update' method in 
        the main simulation loop.
        """

        #iterate all possible starting blocks (nodes of directed graph)
        max_path_length = 0
        for block in self.blocks:

            #recursively compute the longest path via depth first search
            path_length = path_length_dfs(self.connections, block)
            if path_length > max_path_length:
                max_path_length = path_length

        #set 'iterations_min' for fixed-point loop if not provided globally
        if self.iterations_min is None:
            self.iterations_min = max(1, max_path_length)

            #logging message, using path length as minimum iterations
            self._logger_info(f"PATH LENGTH ESTIMATE {max_path_length}, 'iterations_min' set to {self.iterations_min}")

        else:
            #logging message
            self._logger_info(f"PATH LENGTH ESTIMATE {max_path_length}")


    # solver management -----------------------------------------------------------

    def _set_solver(self, Solver=None, **solver_args):
        """
        Initialize all blocks with solver for numerical integration
        and tolerance for local truncation error 'tolerance_lte'.

        If blocks already have solvers, change the numerical integrator
        to the 'Solver' class.

        INPUTS:
            Solver      : ('Solver' class) numerical solver definition
            solver_args : (dict) additional parameters for numerical solvers such as abs and rel tolerance
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
        self._logger_info(f"SOLVER {self.engine} adaptive={self.engine.is_adaptive} implicit={not self.engine.is_explicit}")


    # resetting -------------------------------------------------------------------

    def reset(self):
        """
        Reset the blocks to their initial state and the global time of 
        the simulation. For recording blocks such as 'Scope', their recorded 
        data is also reset. 

        Afterwards the system function os evaluated with '_update' to update
        the block inputs and outputs.
        """

        self._logger_info("RESET")

        #reset simulation time
        self.time = 0.0

        #reset blocks to initial state
        for block in self.blocks:
            block.reset()

        #evaluate system function
        self._update(0.0)


    # timestepping ----------------------------------------------------------------

    def _revert(self):
        """
        Revert simulation state to previous timestep for adaptive solvers 
        when local truncation error is too large and timestep has to be 
        retaken with smaller timestep.
        """
        for block in self.blocks:
            block.revert()


    def _sample(self, t):
        """
        Sample data from blocks that implement the 'sample' method such 
        as 'Scope', 'Delay' and the blocks that sample from a random 
        distribution at a given time 't'.

        INPUTS:
            t : (float) time where to sample
        """
        for block in self.blocks:
            block.sample(t)


    def _update(self, t):
        """
        Fixed-point iterations to resolve algebraic loops and distribute 
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

        INPUTS:
            t : (float) evaluation time of the system function
        """

        #perform minimum number of fixed-point iterations without error checking
        for _iteration in range(self.iterations_min):
                        
            #update connenctions (data transfer)
            for connection in self.connections:
                connection.update()

            #update all blocks
            for block in self.blocks:
                block.update(t)

        #perform fixed-point iterations until convergence with error checking
        for iteration in range(self.iterations_min, self.iterations_max):
                        
            #update connenctions (data transfer)
            for connection in self.connections:
                connection.update()

            #update instant time blocks
            max_error = 0.0
            for block in self.blocks:
                error = block.update(t)
                if error > max_error:
                    max_error = error

            #return number of iterations if converged
            if max_error <= self.tolerance_fpi:
                return iteration+1

        #not converged
        _msg = f"fixed-point loop in '_update' not converged, iter={iteration+1}, err={max_error}"
        self._logger_error(_msg)
        raise RuntimeError(_msg)


    def _solve(self, t, dt):
        """
        For implicit solvers, this method implements the solving step 
        of the implicit update equation.

        It already involves the evaluation of the system equation with 
        the '_update' method within the loop.

        This also tracks the evolution of the solution as an estimate 
        for the convergence via the max residual norm of the fixed point 
        equation of the previous solution.

        INPUTS: 
            t  : (float) evaluation time of dynamical timestepping
            dt : (float) timestep

        RETURNS: 
            success                 : (bool) indicator if the timestep was successful
            total_evals       : (int) total number of system evaluations
            total_solver_its : (int) total number of implicit solver iterations
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
                error = block.solve(t, dt)
                if error > max_error:
                    max_error = error

            #check for convergence (only error)
            if max_error <= self.tolerance_fpi:
                return True, total_evals, iteration + 1

        #not converged in 'self.iterations_max' steps
        return False, total_evals, iteration + 1


    def _step(self, t, dt):
        """
        Performs the 'step' method for dynamical blocks with internal 
        states that have a numerical integration engine. 
        Collects the local truncation error estimates and the timestep 
        rescale factor from the error controllers of the internal 
        intergation engines if they provide an error estimate 
        (for example embedded Runge-Kutta methods).
        
        NOTE: 
            Not to be confused with the global 'step' method, the '_step' 
            method executes the intermediate timesteps in multistage solvers 
            such as Runge-Kutta methods.

        INPUTS: 
            t  : (float) evaluation time of dynamical timestepping
            dt : (float) timestep

        RETURNS: 
            success   : (bool) indicator if the timestep was successful
            max_error : (float) maximum local truncation error from integration
            scale     : (float) rescale factor for timestep
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

        #calculate real relevant timestep rescale
        if not relevant_scales: scale = 1.0  
        else: scale = min(relevant_scales)

        return success, max_error_norm, scale


    def step(self, dt=None, adaptive=False):
        """
        Advances the simulation by one timestep 'dt'. 

        If the 'adaptive' flag is set to 'True' and the selected solver 
        supports adaptive timestepping ('self.is_adaptive'), and the 
        local truncation error or the solver error exceeds the tolerance 
        'tolerance_lte', simulation state is reverted ('revert') to the 
        state before the 'step' method was called.

        INPUTS: 
            dt       : (float) timestep
            adaptive : (bool) use adaptive timestepping (if available)

        RETURNS:
            success                 : (bool) indicator if the timestep was successful
            max_error               : (float) maximum local truncation error from integration
            scale                   : (float) rescale factor for timestep
            total_evals       : (int) total number of system evaluations
            total_solver_its : (int) total number of implicit solver iterations
        """

        #default global timestep as local timestep
        if dt is None: 
            dt = self.dt

        #buffer internal states
        for block in self.blocks:
            block.buffer(dt)

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
                if adaptive and not success:
                    error_norm, scale = 0.0, 0.5
                    break    

            #timestep for dynamical blocks (with internal states)
            success, error_norm, scale = self._step(time, dt)

        #if step not successful and adaptive -> quit early
        if adaptive and not success:
            self._revert()
            return success, error_norm, scale, total_evals, total_solver_its
        
        #increment global time and continue simulation
        self.time += dt 
        
        #evaluate system equation before recording state
        total_evals += self._update(self.time) 

        #sample data after successful timestep
        self._sample(self.time)

        #max local truncation error, timestep rescale, successful step
        return success, error_norm, scale, total_evals, total_solver_its


    def run(self, duration=10, reset=True):
        """
        Perform multiple simulation timesteps for a given 'duration' in seconds.

        INPUTS: 
            duration : (float) simulation time (in time units)
            reset    : (bool) reset the simulation before running

        RETURN:
            steps                   : (int) total number of simulation timesteps
            total_evals       : (int) total number of system evaluations
            total_solver_its : (int) total number of implicit solver iterations
        """

        #reset the simulation before running it
        if reset:
            self.reset()

        #log message solver selection
        self._logger_info(f"RUN duration={duration}")

        #simulation start and end time
        start_time, end_time = self.time, self.time + duration

        #effective timestep for duration
        _dt = self.dt

        #initialize progress tracker
        tracker = ProgressTracker(logger=self.logger, log_interval=10)

        #count the number of function evaluations and solver iterations
        total_evals, total_solver_its = 0, 0
        
        #initial system function evaluation 
        total_evals += self._update(self.time)

        #sampling states and inputs at 'self.time == starting_time' 
        self._sample(self.time)

        #iterate progress tracker generator until 'progress >= 1.0' is reached
        for _ in tracker:

            #rescale effective timestep if in danger of overshooting 'end_time'
            if self.time + _dt > end_time:
                _dt = end_time - self.time

            #advance the simulation by one (effective) timestep '_dt'
            success, _, scale, evals, solver_its = self.step(_dt, self.engine.is_adaptive)

            #update evaluation and iteration counters
            total_evals += evals
            total_solver_its += solver_its

            #rescale the timestep for error control if adaptive solver 
            if self.engine.is_adaptive:

                #apply bounds to timestep
                _dt = np.clip(scale*_dt, self.dt_min, self.dt_max)

            #calculate progress and update progress tracker
            progress = (self.time - start_time)/duration
            tracker.check(progress, success)

        return tracker.steps, total_evals, total_solver_its