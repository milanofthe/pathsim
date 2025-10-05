########################################################################################
##
##                          Testing nonlinear duffing oscillator
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection
from pathsim.blocks import (
    ODE,
    Integrator,
    Amplifier,
    Source,
    Function,
    Adder,
    Scope
    )

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54,
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestDuffingSystem(unittest.TestCase):
    """Duffing oscillator built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #initial position and velocity
        x0, v0 = 0.0, 0.0

        #driving angular frequency and amplitude
        self.a, self.omega = 5.0, 2.0

        #parameters (mass, damping, linear stiffness, nonlinear stiffness)
        self.m, self.c, self.k, self.d = 1.0, 0.5, 1.0, 1.4

        #blocks that define the system
        I1 = Integrator(v0)                      # integrator for velocity
        I2 = Integrator(x0)                      # integrator for position
        A1 = Amplifier(self.c)
        A2 = Amplifier(self.k)
        A3 = Amplifier(-1/self.m)
        F1 = Function(lambda x: self.d*x**3)     # nonlinear stiffness
        Sr = Source(lambda t: self.a*np.sin(self.omega*t))
        P1 = Adder()
        self.Sco = Scope(labels=["velocity", "position"])

        blocks = [I1, I2, A1, A2, A3, P1, F1, Sr, self.Sco]

        #connections between the blocks
        connections = [
            Connection(I1, I2, A1, self.Sco),
            Connection(I2, F1),
            Connection(F1, P1[2]),
            Connection(Sr, P1[3]),
            Connection(I2, A2, self.Sco[1]),
            Connection(A1, P1),
            Connection(A2, P1[1]),
            Connection(P1, A3),
            Connection(A3, I1)
            ]

        #initialize simulation
        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )

        # build reference solution with scipy integrator
        xv_0 = np.array([v0, x0])
        def f_duffing(t, _x):
            v, x = _x
            force = self.a*np.sin(self.omega*t)
            return np.array([(-self.c*v - self.k*x - self.d*x**3 + force)/self.m, v])
        sol = solve_ivp(f_duffing, [0, 50], xv_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
        self._reference = sol.sol


    def test_eval_solvers(self):

        #test for different solvers
        for SOL in [RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKDP87]:

            #test for different tolerances
            for tol in [1e-5, 1e-6, 1e-7]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL,
                        tolerance_lte_rel=0.0,
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(50)

                    time, [v, x] = self.Sco.read()
                    vr, xr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(v-vr)), tol, 2)


class TestDuffingODE(unittest.TestCase):
    """Duffing oscillator as an ODE block"""

    def setUp(self):

        #initial position and velocity
        v0, x0 = 0.0, 0.0

        #driving angular frequency and amplitude
        self.a, self.omega = 5.0, 2.0

        #parameters (mass, damping, linear stiffness, nonlinear stiffness)
        self.m, self.c, self.k, self.d = 1.0, 0.5, 1.0, 1.4

        xv_0 = np.array([v0, x0])

        def f_duffing(_x, u, t):
            v, x = _x
            force = self.a*np.sin(self.omega*t)
            return np.array([(-self.c*v - self.k*x - self.d*x**3 + force)/self.m, v])

        duf = ODE(f_duffing, initial_value=xv_0)

        self.sco = Scope(labels=["velocity", "position"])

        self.Sim = Simulation(
            blocks=[duf, self.sco],
            connections=[
                Connection(duf[0], self.sco[0]),
                Connection(duf[1], self.sco[1])
                ],
            log=False
            )

        # build reference solution with scipy integrator
        def f_duffing_(t, _x):
            v, x = _x
            force = self.a*np.sin(self.omega*t)
            return np.array([(-self.c*v - self.k*x - self.d*x**3 + force)/self.m, v])
        sol = solve_ivp(f_duffing_, [0, 50], xv_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
        self._reference = sol.sol


    def test_eval_solvers(self):

        #test for different solvers
        for SOL in [RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKDP87]:

            #test for different tolerances
            for tol in [1e-5, 1e-6, 1e-7]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL,
                        tolerance_lte_rel=0.0,
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(50)

                    time, [v, x] = self.sco.read()
                    vr, xr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(v-vr)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
