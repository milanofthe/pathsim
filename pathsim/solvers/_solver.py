########################################################################################
##
##                            BASE NUMERICAL INTEGRATOR CLASSES
##                                  (solvers/_solver.py)
##
##                                  Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ..optim.anderson import (
    Anderson, 
    NewtonAnderson
    )

from ..optim.newton import (
    LevenbergMarquardtAD,
    NewtonRaphsonAD, 
    GaussNewtonAD
    )


# BASE SOLVER CLASS ====================================================================

class Solver:
    """Base skeleton class for solver definition. Defines the basic solver methods and 
    the metadata.

    Specific solvers need to implement (some of) the base class methods defined here. 
    This depends on the type of solver (implicit/explicit, multistage, adaptive).

    Parameters
    ----------
    initial_value : float, array
        initial condition / integration constant
    func : callable
        function to integrate with state 'x', input 'u' and time 't' dependency
    jac : callable, None
        jacobian of 'func' with respect to 'x', depending on 'x', 'u' and 't', 
        if 'None', no jacobian is used
    tolerance_lte_abs : float
        absolute tolerance for local truncation error (for solvers with error estimate)
    tolerance_lte_rel : float
        relative tolerance for local truncation error (for solvers with error estimate)

    Attributes
    ----------
    x_0 : numeric, array[numeric]
        internal 'working' initial value
    x : numeric, array[numeric]
        internal 'working' state
    n : int
        order of integration scheme
    s : int
        number of internal intermediate stages
    stage : int
        counter for current intermediate stage
    eval_stages : list[float]
        rations for evaluation times of intermediate stages
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-8, 
                 tolerance_lte_rel=1e-5):

        #set buffer, state and initial condition    
        self.x_0 = self.x = self.initial_value = initial_value

        #right hand side function for integration
        self.func = func

        #jacobian of right hand side function
        self.jac = jac

        #tolerances for local truncation error (for adaptive solvers)
        self.tolerance_lte_abs = tolerance_lte_abs  
        self.tolerance_lte_rel = tolerance_lte_rel  

        #flag to identify adaptive/fixed timestep solvers
        self.is_adaptive = False

        #order of the integration scheme
        self.n = 1

        #number of stages
        self.s = 1

        #current evaluation stage for multistage solvers
        self.stage = 0

        #intermediate evaluation times as ratios between [t, t+dt]
        self.eval_stages = [0.0]


    def __str__(self):
        return self.__class__.__name__


    def __len__(self):
        """size of the internal state, i.e. the order
        
        Returns
        -------
        size : int
            size of the current internal state
        """
        return len(np.atleast_1d(self.x))


    def __bool__(self):
        return True


    def stages(self, t, dt):
        """Generator that yields the intermediate evaluation 
        time during the timestep 't + ratio * dt'.

        Parameters
        ----------
        t : float 
            evaluation time
        dt : float
            integration timestep
        """
        for ratio in self.eval_stages:
            yield t + ratio * dt


    def get(self):
        """Returns current internal state of the solver.
    
        Returns
        -------
        x : numeric, array[numeric]
            current internal state of the solver
        """
        return self.x

    
    def set(self, x):
        """Sets the internal state of the integration engine.

        This method is required for event based simulations, 
        and to handle discontinuities in state variables.
        
        Parameters
        ----------
        x : numeric, array[numeric]
            new internal state of the solver

        """

        #overwrite internal state with value
        self.x = self.x_0 = x

        #reset stage counter
        self.stage = 0


    def reset(self):
        """"Resets integration engine to initial value"""

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value

        #reset stage counter
        self.stage = 0


    def buffer(self, dt):
        """Saves the current state to an internal state buffer which 
        is especially relevant for multistage and implicit solvers.

        Multistep solver implement rolling buffers for the states 
        and timesteps.

        Resets the stage counter.
        
        Parameters
        ----------
        dt : float
            integration timestep
    
        """

        #buffer internal state
        self.x_0 = self.x

        #reset stage counter
        self.stage = 0


    @classmethod
    def cast(cls, other, **solver_args):
        """Cast the integration engine to the new type and initialize 
        with previous solver arguments so it can continue from where 
        the 'old' solver stopped.
            
        Parameters
        ----------
        other : Solver
            solver instance to cast to new solver type
        solver_args : dict
            additional args for the new solver

        Returns
        -------
        engine : Solver
            new solver instance        
        """

        if not isinstance(other, Solver):
            raise ValueError("'other' must be instance of 'Solver' or child")


        #create new solver instance
        engine = cls(
            initial_value=other.initial_value, 
            func=other.func, 
            jac=other.jac, 
            tolerance_lte_rel=solver_args.get("tolerance_lte_rel", other.tolerance_lte_rel),
            tolerance_lte_abs=solver_args.get("tolerance_lte_abs", other.tolerance_lte_abs)
            )
        
        #set internal state of new engine from other
        engine.set(other.get())

        return engine


    # methods for adaptive timestep solvers --------------------------------------------

    def error_controller(self):
        """Returns the estimated local truncation error (abs and rel) and scaling factor 
        for the timestep, only relevant for adaptive timestepping methods.

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """
        return True, 0.0, 1.0


    def revert(self):
        """Revert integration engine to previous timestep. 

        This is only relevant for adaptive methods where the simulation 
        timestep 'dt' is rescaled and the engine step is recomputed with 
        the smaller timestep.
        """
        
        #reset internal state to previous state
        self.x = self.x_0

        #reset stage counter
        self.stage = 0   


    # methods for timestepping ---------------------------------------------------------

    def step(self, u, t, dt):
        """Performs the explicit timestep for (t+dt) based 
        on the state and input at (t).

        Returns the local truncation error estimate and the 
        rescale factor for the timestep if the solver is adaptive.

        Parameters
        ----------
        u : numeric, array[numeric]
            function 'func' input value
        t : float
            evaluation time of function 'func'
        dt : float 
            integration timestep

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """
        return True, 0.0, 1.0


    # methods for interpolation --------------------------------------------------------

    def interpolate(self, r, dt):
        """Interpolate solution after successful timestep as a ratio 
        in the interval [t, t+dt].

        This is especially relevant for Runge-Kutta solvers that 
        have a higher order interpolant. Otherwise this is just 
        linear interpolation using the buffered state.
        
        Parameters
        ----------
        r : float
            ration for interpolation within timestep
        dt : float
            integration timestep

        Returns
        -------
        x : numeric, array[numeric]
            interpolated state
        """
        _r = np.clip(r, 0.0, 1.0)
        return _r * self.x + (1.0 - _r) * self.x_0


# EXTENDED BASE SOLVER CLASSES =========================================================

class ExplicitSolver(Solver):
    """Base class for explicit solver definition.

    Attributes
    ----------
    x_0 : numeric, array[numeric]
        internal 'working' initial value
    x : numeric, array[numeric]
        internal 'working' state
    n : int
        order of integration scheme
    s : int
        number of internal intermediate stages
    stage : int
        counter for current intermediate stage
    eval_stages : list[float]
        rations for evaluation times of intermediate stages

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #flag to identify implicit/explicit solvers
        self.is_explicit = True
        self.is_implicit = False

        #intermediate evaluation times for multistage solvers as ratios between [t, t+dt]
        self.eval_stages = [0.0]


    # method for direct integration ----------------------------------------------------

    def integrate_singlestep(self, time=0.0, dt=0.1):
        """Directly integrate the function 'func' for a single timestep 'dt' with 
        explicit solvers. This method is primarily intended for testing purposes.
        
        Parameters
        ----------  
        time_start : float
            starting time for timestep
        dt : float
            integration timestep

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error_norm : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #buffer current state
        self.buffer(dt)

        #iterate solver stages (explicit updates)
        for t in self.stages(time, dt):
            success, error_norm, scale = self.step(0.0, t, dt)

        return success, error_norm, scale


    def integrate(self, 
                  time_start=0.0, 
                  time_end=1.0, 
                  dt=0.1, 
                  dt_min=0.0, 
                  dt_max=None, 
                  adaptive=True):
        """Directly integrate the function 'func' from 'time_start' 
        to 'time_end' with timestep 'dt' for explicit solvers. 

        This method is primarily intended for testing purposes.
    
        Parameters
        ----------
        time_start : float
            starting time for integration
        time_end : float
            end time for integration
        dt : float
            timestep or initial timestep for adaptive solvers
        dt_min : float
            lower bound for timestep, default '0.0'
        dt_max : float
            upper bound for timestep, default 'None'
        adaptive : bool
            use adaptive timestepping if available

        Returns
        -------
        outout_times : array[float]
            time points of the solution
        output_states : array[numeric], array[array[numeric]]
            state values at solution time points
        """

        #output lists with initial state
        output_states = [self.x]
        output_times = [time_start]

        #integration starting time
        time = time_start

        #step until duration is reached
        while time < time_end + dt:

            #perform single timestep
            success, error_norm, scale = self.integrate_singlestep(time, dt)

            #check if timestep was successful
            if adaptive and not success:
                self.revert()
            else:
                time += dt
                output_states.append(self.x)
                output_times.append(time)

            #rescale and apply bounds to timestep
            if adaptive:
                if scale*dt < dt_min:
                    raise RuntimeError("Error control requires timestep smaller 'dt_min'!")
                dt = np.clip(scale*dt, dt_min, dt_max)

        #return the evaluation times and the states
        return np.array(output_times), np.array(output_states)


class ImplicitSolver(Solver):
    """
    Base class for implicit solver definition.

    Attributes
    ----------
    x_0 : numeric, array[numeric]
        internal 'working' initial value
    x : numeric, array[numeric]
        internal 'working' state
    n : int
        order of integration scheme
    s : int
        number of internal intermediate stages
    stage : int
        counter for current intermediate stage
    eval_stages : list[float]
        rations for evaluation times of intermediate stages
    opt : NewtonAnderson, Anderson, etc.
        optimizer instance to solve the implicit update equation

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #flag to identify implicit/explicit solvers
        self.is_explicit = False
        self.is_implicit = True

        #intermediate evaluation times for multistage solvers as ratios between [t, t+dt]
        self.eval_stages = [1.0]

        #initialize optimizer for solving implicit update equation
        # self.opt = Anderson(m=5, restart=False)
        self.opt = NewtonAnderson(m=5, restart=False)
        # self.opt = NewtonRaphsonAD()
        # self.opt = GaussNewtonAD()
        # self.opt = LevenbergMarquardtAD()


    def buffer(self, dt):
        """Saves the current state to an internal state buffer which 
        is especially relevant for multistage and implicit solvers.

        Resets the stage counter and the optimizer of implicit methods.
        
        Parameters
        ----------
        dt : float
            integration timestep
        """

        #buffer internal state
        self.x_0 = self.x

        #reset stage counter
        self.stage = 0

        #reset optimizer
        self.opt.reset()


    # methods for timestepping ---------------------------------------------------------

    def solve(self, u, t, dt):
        """Advances the solution of the implicit update equation of the solver 
        with the optimizer of the engine and tracks the evolution of the 
        solution by providing the residual norm of the fixed-point solution.

        Parameters
        ----------
        u : numeric, array[numeric]
            function 'func' input value
        t : float
            evaluation time of function 'func'
        dt : float 
            integration timestep

        Returns
        -------
        err : float
            residual error of the fixed point update equation
        """
        return 0.0


    # method for direct integration ----------------------------------------------------

    def integrate_singlestep(self, 
                             time=0.0, 
                             dt=0.1, 
                             tolerance_fpi=1e-12, 
                             max_iterations=5000):
        """
        Directly integrate the function 'func' for a single timestep 'dt' with 
        implicit solvers. This method is primarily intended for testing purposes.

        Parameters
        ----------  
        time_start : float
            starting time for timestep
        dt : float
            integration timestep
        tolerance_fpi : float
            convergence criterion for implicit update equation
        max_iterations : int
            maximum numer of iterations for optimizer to solve 
            implicit update equation

        Returns 
        -------
        success : bool
            True if the timestep was successful
        success_sol : bool
            True if optimizer successfully solved implicit update equation
        error_norm : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #buffer current state
        self.buffer(dt)

        #flag for solver success
        success_sol = True

        #iterate solver stages (implicit updates)
        for t in self.stages(time, dt):
            
            #iteratively solve implicit update equation
            for _ in range(max_iterations):
                error_sol = self.solve(0.0, t, dt)
                if error_sol < tolerance_fpi: 
                    break

            #catch convergence error 
            if error_sol > tolerance_fpi:
                if success_sol: success_sol = False
            
            #perform explicit component of timestep
            success, error_norm, scale = self.step(0.0, t, dt)

        return success, success_sol, error_norm, scale 


    def integrate(self, 
                  time_start=0.0, 
                  time_end=1.0, 
                  dt=0.1, 
                  dt_min=0.0, 
                  dt_max=None, 
                  adaptive=True,
                  tolerance_fpi=1e-12, 
                  max_iterations=5000):
        """Directly integrate the function 'func' from 'time_start' to 'time_end' with 
        timestep 'dt' for implicit solvers. 

        This method is primarily intended for testing purposes.
    
        Parameters
        ----------
        time_start : float
            starting time for integration
        time_end : float
            end time for integration
        dt : float
            timestep or initial timestep for adaptive solvers
        dt_min : float
            lower bound for timestep, default '0.0'
        dt_max : float
            upper bound for timestep, default 'None'
        adaptive : bool
            use adaptive timestepping if available
        tolerance_fpi : float
            convergence criterion for implicit update equation
        max_iterations : int
            maximum numer of iterations for optimizer to solve 
            implicit update equation

        Returns
        -------
        outout_times : array[float]
            time points of the solution
        output_states : array[numeric], array[array[numeric]]
            state values at solution time points    
        """

        #output lists with initial state
        output_states = [self.x]
        output_times = [time_start]

        #integration starting time
        time = time_start

        #step until duration is reached
        while time < time_end + dt:

            #integrate for single timestep
            success, success_sol, error_norm, scale = self.integrate_singlestep(
                time, 
                dt, 
                tolerance_fpi, 
                max_iterations
                )

            #check if timestep was successful and adaptive
            if adaptive and not (success and success_sol):
                self.revert()
            else:
                time += dt
                output_states.append(self.x)
                output_times.append(time)

            #rescale and apply bounds to timestep
            if adaptive:
                if scale*dt < dt_min:
                    raise RuntimeError("Error control requires timestep smaller 'dt_min'!")
                dt = np.clip(scale*dt, dt_min, dt_max)

        #return the evaluation times and the states
        return np.array(output_times), np.array(output_states)