########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk54.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ImplicitSolver
from ..utils.funcs import numerical_jacobian

# SOLVERS ==============================================================================

class ESDIRK54(ImplicitSolver):
    """
    7 stage 5-th order L-stable and stiffly accurate ESDIRK method with 
    embedded 4-th order method for stepsize control. This integrator is 
    suited for moderately stiff problems that require high accuracy.
    The first stage is explicit, followed by 6 implicit stages.

    FROM : 
        Diagonally implicit Runge–Kutta methods for stiff ODEs
        Christopher A.Kennedy, Mark H.Carpenter
        Applied Numerical Mathematics, 2019
        ESDIRK5(4)7L[2]SA2
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

        #counter for runge kutta stages
        self.stage = 0

        #flag adaptive timestep solver
        self.is_adaptive = True

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [0.0, 46/125, 7121331996143/11335814405378, 49/353, 3706679970760/5295570149437, 347/382, 1.0]

        #butcher table
        self.BT = {0:[0.0],
                   1:[23/125, 23/125], 
                   2:[791020047304/3561426431547, 791020047304/3561426431547, 23/125], 
                   3:[-158159076358/11257294102345, -158159076358/11257294102345, -85517644447/5003708988389, 23/125], 
                   4:[-1653327111580/4048416487981, -1653327111580/4048416487981, 1514767744496/9099671765375, 14283835447591/12247432691556, 23/125],
                   5:[-4540011970825/8418487046959, -4540011970825/8418487046959, -1790937573418/7393406387169, 10819093665085/7266595846747, 4109463131231/7386972500302, 23/125],
                   6:[-188593204321/4778616380481, -188593204321/4778616380481, 2809310203510/10304234040467, 1021729336898/2364210264653, 870612361811/2470410392208, -1307970675534/8059683598661, 23/125]}

        #coefficients for truncation error estimate
        _A1 = [-188593204321/4778616380481, -188593204321/4778616380481, 2809310203510/10304234040467, 1021729336898/2364210264653, 870612361811/2470410392208, -1307970675534/8059683598661, 23/125]
        _A2 = [-582099335757/7214068459310, -582099335757/7214068459310, 615023338567/3362626566945, 3192122436311/6174152374399, 6156034052041/14430468657929, -1011318518279/9693750372484, 1914490192573/13754262428401]
        self.TR = [_a1 - _a2 for _a1, _a2 in zip(_A1, _A2)]


    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep 
        based on local truncation error estimate and returns both
        """
        if len(self.Ks)<len(self.TR): 
            return True, 0.0, 0.0, 1.0

        #compute local truncation error
        tr = dt * sum(k*b for k, b in zip(self.Ks.values(), self.TR))

        #compute and clip truncation error, error ratio abs
        truncation_error_abs = float(np.max(np.clip(abs(tr), 1e-18, None)))
        error_ratio_abs = self.tolerance_lte_abs / truncation_error_abs

        #compute and clip truncation error, error ratio rel
        if np.any(self.x == 0.0): 
            truncation_error_rel = 1.0
            error_ratio_rel = 0.0
        else:
            truncation_error_rel = float(np.max(np.clip(abs(tr/self.x), 1e-18, None)))
            error_ratio_rel = self.tolerance_lte_rel / truncation_error_rel
        
        #compute error ratio and success check
        error_ratio = max(error_ratio_abs, error_ratio_rel)
        success = error_ratio >= 1.0

        #compute timestep scale
        timestep_rescale = 0.9 * (error_ratio)**(1/5)

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #first stage is explicit
        if self.stage == 0:
            return 0.0
            
        #update timestep weighted slope 
        self.Ks[self.stage] = self.func(self.x, u, t)

        #compute slope and update fixed-point equation
        slope = sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage]))

        #use the jacobian
        if self.jac is not None:

            #most recent butcher coefficient
            b = self.BT[self.stage][self.stage]

            #compute jacobian of fixed-point equation
            jac_g = dt * b * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        performs the timestep update
        """

        #first stage is explicit
        if self.stage == 0:
            self.Ks[self.stage] = self.func(self.x, u, t)

        #restart anderson accelerator 
        self.acc.reset()

        #error and step size control
        if self.stage < 6:
            self.stage += 1
            return True, 0.0, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)