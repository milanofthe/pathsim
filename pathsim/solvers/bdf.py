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
from ..utils.funcs import numerical_jacobian


# SOLVERS ==============================================================================

class BDF2(ImplicitSolver):
    """
    2nd order backward differentiation formula 
    with order ramp up for the initial steps.
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

        #bdf coefficients
        self.K = {1:[1.0], 2:[-1/3, 4/3]}
        self.F = {1:1.0, 2:2/3}

        #bdf solution buffer
        self.B = [self.initial_value]


    def reset(self):
        """"
        Resets integration engine to initial state.
        """

        #reset buffer state with initial value
        self.B = [self.initial_value]

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #buffer length for BDF order selection
        n = len(self.B)

        #fixed-point function update
        g = self.F[n] * dt * self.func(self.x, u, t) + sum(b*k for b, k in zip(self.B, self.K[n]))

        #use the jacobian
        if self.jac is not None:

            #compute jacobian
            jac_g = self.F[n] * dt * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, g, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        Performs the timestep by buffereing the previous state.
        """

        #reset anderson accelerator
        self.acc.reset()

        #add to buffer
        self.B.append(self.x)
        if len(self.B) > 2:
            self.B.pop(0)

        return True, 0.0, 0.0, 1.0


class BDF3(ImplicitSolver):
    """
    3rd order backward differentiation formula 
    with order ramp up for the initial steps.
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

        #bdf coefficients
        self.K = {1:[1.0], 
                  2:[-1/3, 4/3], 
                  3:[2/11, -9/11, 18/11]}
        self.F = {1:1.0, 2:2/3, 3:6/11}

        #bdf solution buffer
        self.B = [self.initial_value]


    def reset(self):
        """"
        Resets integration engine to initial state.
        """

        #reset buffer state with initial value
        self.B = [self.initial_value]

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #buffer length for BDF order selection
        n = len(self.B)

        #fixed-point function update
        g = self.F[n] * dt * self.func(self.x, u, t) + sum(b*k for b, k in zip(self.B, self.K[n]))

        #use the jacobian
        if self.jac is not None:

            #compute jacobian
            jac_g = self.F[n] * dt * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, g, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, g, None)

        #return the fixed-point residual
        return err



    def step(self, u, t, dt):
        """
        Performs the timestep by buffereing the previous state.
        """

        #reset anderson accelerator
        self.acc.reset()

        #add to buffer
        self.B.append(self.x)
        if len(self.B) > 3:
            self.B.pop(0)

        return True, 0.0, 0.0, 1.0


class BDF4(ImplicitSolver):
    """
    4th order backward differentiation formula 
    with order ramp up for the initial steps.
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

        #bdf coefficients
        self.K = {1:[1.0], 
                  2:[-1/3, 4/3], 
                  3:[2/11, -9/11, 18/11], 
                  4:[-3/25, 16/25, -36/25, 48/25]}
        self.F = {1:1.0, 2:2/3, 3:6/11, 4:12/25}

        #bdf solution buffer
        self.B = [self.initial_value]


    def reset(self):
        """"
        Resets integration engine to initial state.
        """

        #reset buffer state with initial value
        self.B = [self.initial_value]

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #buffer length for BDF order selection
        n = len(self.B)

        #fixed-point function update
        g = self.F[n] * dt * self.func(self.x, u, t) + sum(b*k for b, k in zip(self.B, self.K[n]))

        #use the jacobian
        if self.jac is not None:

            #compute jacobian
            jac_g = self.F[n] * dt * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, g, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        Performs the timestep by buffereing the previous state.
        """

        #reset anderson accelerator
        self.acc.reset()

        #add to buffer
        self.B.append(self.x)
        if len(self.B) > 4:
            self.B.pop(0)

        return True, 0.0, 0.0, 1.0