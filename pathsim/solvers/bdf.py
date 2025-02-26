########################################################################################
##
##                         BACKWARD DIFFERENTIATION FORMULAS
##                                 (solvers/bdf.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ImplicitSolver


# BASE BDF SOLVER ======================================================================

class BDF(ImplicitSolver):
    """Base class for the backward differentiation formula (BDF) integrators.
    
    Notes
    ----- 
    This solver class is not intended to be used directly

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
    K : dict[int: list[float]]
        bdf coefficients for the state buffer for each order
    F : dict[int: float]
        bdf coefficients for the function 'func' for each order
    B : list[numeric], list[array[numeric]]
        buffer for previous states
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order
        self.n = None

        #bdf coefficients for orders 1 to 6
        self.K = {1:[1.0], 
                  2:[-1/3, 4/3], 
                  3:[2/11, -9/11, 18/11], 
                  4:[-3/25, 16/25, -36/25, 48/25],
                  5:[12/137, -75/137, 200/137, -300/137, 300/137],
                  6:[-10/147, 72/147, -225/147, 400/147, -450/147, 360/147]}
        self.F = {1:1.0, 2:2/3, 3:6/11, 4:12/25, 5:60/137, 6:60/147}

        #bdf solution buffer
        self.B = []


    def reset(self):
        """"Resets integration engine to initial state."""

        #clear buffer
        self.B = []

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value


    def buffer(self, dt):
        """buffer the state for the multistep method
        
        Parameters
        ----------
        dt : float
            integration timestep

        """
            
        #reset optimizer
        self.opt.reset()

        #buffer state directly
        self.x_0 = self.x

        #add to buffer
        self.B.append(self.x)

        #truncate buffer if too long
        if len(self.B) > self.n:
            self.B.pop(0)


    def solve(self, u, t, dt):
        """Solves the implicit update equation using the optimizer of the engine.

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

        #buffer length for BDF order selection
        n = min(len(self.B), self.n)

        #fixed-point function update
        g = self.F[n] * dt * self.func(self.x, u, t) 
        for b, k in zip(self.B, self.K[n]):
            g = g + b*k

        #use the jacobian
        if self.jac is not None:

            #compute jacobian
            jac_g = self.F[n] * dt * self.jac(self.x, u, t)

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, jac_g)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err


# SOLVERS ==============================================================================

class BDF2(BDF):
    """2-nd order backward differentiation formula 
    with order ramp up for the initial steps.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 2


class BDF3(BDF):
    """3-rd order backward differentiation formula 
    with order ramp up for the initial steps.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 3


class BDF4(BDF):
    """4-th order backward differentiation formula 
    with order ramp up for the initial steps.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 4


class BDF5(BDF):
    """5-th order backward differentiation formula 
    with order ramp up for the initial steps.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 5


class BDF6(BDF):
    """6-th order backward differentiation formula 
    with order ramp up for the initial steps.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 6