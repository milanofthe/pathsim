#########################################################################################
##
##                                 CONTROL BLOCKS
##                                (blocks/ctrl.py)
##
##                                Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import DynamicOperator


# SISO BLOCKS ===========================================================================

class PID(Block):
    """Proportional-Integral-Differntiation (PID) controller.

    The transfer function is defined as

    .. math::
        
        H_\\mathrm{diff}(s) = K_p + K_i \\frac{1}{s} + K_d \\frac{s}{1 + s / f_\\mathrm{max}} 

    where the differentiation is approximated by a high pass filter that holds 
    for signals up to a frequency of approximately f_max.

    Note
    ----
    Depending on 'f_max', the resulting system might become stiff or ill conditioned!
    As a practical choice set f_max to 3x the highest expected signal frequency.
    
    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #cutoff at 1kHz
        pid = PID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3)

    Parameters
    ----------
    Kp : float
        poroportional controller coefficient
    Ki : float
        integral controller coefficient
    Kd : float
        differentiator controller coefficient
    f_max : float
        highest expected signal frequency

    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for ODE component
    op_alg : DynamicOperator
        internal algebraic operator

    """

    def __init__(self, Kp=0, Ki=0, Kd=0, f_max=100):
        super().__init__()

        #pid controller coefficients
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        #maximum frequency for differentiator approximation
        self.f_max = f_max

        self.op_dyn = DynamicOperator(
            func=lambda x, u, t: np.array([self.f_max * (u - x[0]), u]),
            )
        self.op_alg = DynamicOperator(
            func=lambda x, u, t: self.Kd * self.f_max * (u - x[0]) + self.Ki * x[1] + self.Kp * u,
            jac_x=lambda x, u, t: np.array([-self.Kd * self.f_max, self.Ki]),
            jac_u=lambda x, u, t: self.Kd * self.f_max + self.Kp,
            )


    def __len__(self):
        return 1 if self._active and (self.Kp or self.Kd) else 0


    def set_solver(self, Solver, **solver_args):
        """set the internal numerical integrator

        Parameters
        ----------
        Solver : Solver
            numerical integration solver class
        solver_args : dict
            parameters for solver initialization
        """
        #initialize the numerical integration engine with kernel
        if not self.engine: self.engine = Solver(np.zeros(2), **solver_args)
        #change solver if already initialized    
        else: self.engine = Solver.cast(self.engine, **solver_args)    


    def update(self, t):
        """update system equation fixed point loop
    
        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
        x, u = self.engine.get(), self.inputs[0]
        _out, self.outputs[0] = self.outputs[0], self.op_alg(x, u, t)
        return abs(_out - self.outputs[0])


    def solve(self, t, dt):
        """advance solution of implicit update equation

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        x, u = self.engine.get(), self.inputs[0]
        f, J = self.op_dyn(x, u, t), self.op_dyn.jac_x(x, u, t)
        return self.engine.solve(f, J, dt)


    def step(self, t, dt):
        """compute update step with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        x, u = self.engine.get(), self.inputs[0]
        f = self.op_dyn(x, u, t)
        return self.engine.step(f, dt)