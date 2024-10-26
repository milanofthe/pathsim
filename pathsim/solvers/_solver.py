########################################################################################
##
##                            BASE NUMERICAL INTEGRATOR CLASSES
##                                  (solvers/_solver.py)
##
##                                (c) Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ..utils.anderson import (
    AndersonAcceleration, 
    NewtonAndersonAcceleration
    )

# BASE SOLVER CLASS ====================================================================

class Solver:
    """
    Base skeleton class for solver definition. Defines the basic solver methods and the metadata.

    Specific solvers need to implement (some of) the base class methods defined here. 
    This depends on the type of solver (implicit/explicit, multistage, adaptive).

    INPUTS : 
        initial_value     : (float or array) initial condition / integration constant
        func              : (callable) function to integrate with state 'x', input 'u' and time 't' dependency
        jac               : (callable or None) jacobian of 'func' with respect to 'x', depending on 'x', 'u' and 't', if 'None', no jacobian is used
        tolerance_lte_abs : (float) absolute tolerance for local truncation error (for solvers with error estimate)
        tolerance_lte_rel : (float) relative tolerance for local truncation error (for solvers with error estimate)
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):

        #set buffer, initial state and initial condition    
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

        #current evaluation stage for multistage solvers
        self.stage = 0

        #intermediate evaluation times for multistage solvers as ratios between [t, t+dt]
        self.eval_stages = [0.0]


    def __str__(self):
        return self.__class__.__name__


    def __len__(self):
        """
        return the size of the internal state, i.e. the order
        """
        return len(self.x)


    def stages(self, t, dt):
        """
        Generator that yields the intermediate evaluation 
        time during the timestep 't + ratio * dt'.
        """
        for ratio in self.eval_stages:
            yield t + ratio * dt


    def get(self):
        """
        Returns current internal state of the solver.
        """
        return self.x

    
    def set(self, x):
        """
        Sets the internal state of the integration engine.
        This method is required for event based simulations, 
        and to handle discontinuities in state variables.
        """

        #overwrite internal state with value
        self.x = self.x_0 = x

        #reset stage counter
        self.stage = 0


    def reset(self):
        """"
        Resets integration engine to initial state.
        """

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value

        #reset stage counter
        self.stage = 0


    def buffer(self):
        """
        Saves the current state to an internal state buffer, which is 
        especially relevant for multistage and implicit solvers.
        """
        self.x_0 = self.x


    def change(self, Sol, **solver_args):
        """
        Change the integration engine to a new type and initialize 
        with previous solver arguments so it can continue from where 
        the 'old' solver stopped.
        """

        #create new engine from self
        engine = Sol(initial_value=self.initial_value, 
                     func=self.func, 
                     jac=self.jac, 
                     **solver_args)
        
        #set internal state of new engine from self
        engine.set(self.get())

        return engine


    # methods for adaptive timestep solvers --------------------------------------------

    def error_controller(self):
        """
        Returns the estimated local truncation error (abs and rel) and scaling factor 
        for the timestep, only relevant for adaptive timestepping methods.
        """
        return True, 0.0, 0.0, 1.0


    def revert(self):
        """
        Revert integration engine to previous timestep, this is only relevant 
        for adaptive methods where the simulation timestep 'dt' is rescaled and 
        the engine step is recomputed with the smaller timestep.
        """
        
        #reset internal state to previous state
        self.x = self.x_0

        #reset stage counter
        self.stage = 0   

    
    # methods for timestepping ---------------------------------------------------------

    def step(self, u, t, dt):
        """
        performs the explicit timestep for (t+dt) based 
        on the state and input at (t)

        returns the local truncation error estimate and the 
        rescale factor for the timestep if the solver is adaptive.
        """
        return True, 0.0, 0.0, 1.0


# EXTENDED BASE SOLVER CLASSES =========================================================

class ExplicitSolver(Solver):
    """
    Base class for explicit solver definition.

    INPUTS : 
        initial_value     : (float or array) initial condition / integration constant
        func              : (callable) function to integrate with state 'x', input 'u' and time 't' dependency
        jac               : (callable or None) jacobian of 'func' with respect to 'x', depending on 'x', 'u' and 't', if 'None', no jacobian is used
        tolerance_lte_abs : (float) absolute tolerance for local truncation error (for solvers with error estimate)
        tolerance_lte_rel : (float) relative tolerance for local truncation error (for solvers with error estimate)
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):
        super().__init__(initial_value, 
                         func, 
                         jac, 
                         tolerance_lte_abs, 
                         tolerance_lte_rel)

        #flag to identify implicit/explicit solvers
        self.is_explicit = True
        self.is_implicit = False

        #intermediate evaluation times for multistage solvers as ratios between [t, t+dt]
        self.eval_stages = [0.0]


    # method for direct integration ----------------------------------------------------

    def integrate_singlestep(self, time=0.0, dt=0.1):
        """
        Directly integrate the function 'func' for a single timestep 'dt' with 
        explicit solvers. This method is primarily intended for testing purposes.

        INPUTS :    
            time_start : (float) starting time for timestep
            dt         : (float) timestep
        """

        #buffer current state
        self.buffer()

        #iterate solver stages (explicit updates)
        for t in self.stages(time, dt):
            success, error_abs, error_rel, scale = self.step(0.0, t, dt)

        return success, error_abs, error_rel, scale 


    def integrate(self, 
                  time_start=0.0, 
                  time_end=1.0, 
                  dt=0.1, 
                  dt_min=0.0, 
                  dt_max=None, 
                  adaptive=True):
        """
        Directly integrate the function 'func' from 'time_start' to 'time_end' with 
        timestep 'dt' for explicit solvers. This method is primarily intended for 
        testing purposes.

        INPUTS : 
            time_start : (float) starting time for integration
            time_end   : (float) end time for integration
            dt         : (float) timestep or initial timestep for adaptive solvers
            dt_min     : (float) lower bound for timestep, default '0.0'
            dt_max     : (float) upper bound for timestep, default 'None'
            adaptive   : (bool) usa adaptive timestepping if available
        """

        #output lists with initial state
        output_states = [self.x]
        output_times = [time_start]

        #integration starting time
        time = time_start

        #step until duration is reached
        while time < time_end + dt:

            #perform single timestep
            success, error_abs, error_rel, scale = self.integrate_singlestep(time, dt)

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

    INPUTS : 
        initial_value     : (float or array) initial condition / integration constant
        func              : (callable) function to integrate with state 'x', input 'u' and time 't' dependency
        jac               : (callable or None) jacobian of 'func' with respect to 'x', depending on 'x', 'u' and 't', if 'None', no jacobian is used
        tolerance_lte_abs : (float) absolute tolerance for local truncation error (for solvers with error estimate)
        tolerance_lte_rel : (float) relative tolerance for local truncation error (for solvers with error estimate)
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):
        super().__init__(initial_value, 
                         func, 
                         jac, 
                         tolerance_lte_abs, 
                         tolerance_lte_rel)

        #flag to identify implicit/explicit solvers
        self.is_explicit = False
        self.is_implicit = True

        #intermediate evaluation times for multistage solvers as ratios between [t, t+dt]
        self.eval_stages = [1.0]

        #initialize anderson accelerator for solving the implicit update equation
        self.acc = NewtonAndersonAcceleration(m=5, restart=False)


    # methods for timestepping ---------------------------------------------------------

    def solve(self, u, t, dt):
        """
        Advances the solution of the implicit update equation of the solver 
        with Anderson Acceleration and tracks the evolution of the solution
        by providing the residual norm of the fixed-point solution.
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

        INPUTS :    
            time_start     : (float) starting time for timestep
            dt             : (float) timestep
            tolerance_fpi  : (float) tolerance for fixed-point solver  
            max_iterations : (int) maximum number of fixed-point solver iterations
        """

        #buffer current state
        self.buffer()

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
            success, error_abs, error_rel, scale = self.step(0.0, t, dt)

        #step successful in total
        success_total = success and success_sol

        return success_total, error_abs, error_rel, scale 


    def integrate(self, 
                  time_start=0.0, 
                  time_end=1.0, 
                  dt=0.1, 
                  dt_min=0.0, 
                  dt_max=None, 
                  adaptive=True,
                  tolerance_fpi=1e-12, 
                  max_iterations=5000):
        """
        Directly integrate the function 'func' from 'time_start' to 'time_end' with 
        timestep 'dt' for implicit solvers. This method is primarily intended for 
        testing purposes.

        INPUTS : 
            time_start     : (float) starting time for integration
            time_end       : (float) end time for integration
            dt             : (float) timestep or initial timestep for adaptive solvers
            dt_min         : (float) lower bound for timestep, default '0.0'
            dt_max         : (float) upper bound for timestep, default 'None'
            adaptive       : (bool) use adaptive timestepping if available
            tolerance_fpi  : (float) tolerance for fixed-point solver  
            max_iterations : (int) maximum number of fixed-point solver iterations
        """

        #output lists with initial state
        output_states = [self.x]
        output_times = [time_start]

        #integration starting time
        time = time_start

        #step until duration is reached
        while time < time_end + dt:

            #integrate for single timestep
            success, error_abs, error_rel, scale = self.integrate_singlestep(time, dt, tolerance_fpi, max_iterations)

            #check if timestep was successful and adaptive
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