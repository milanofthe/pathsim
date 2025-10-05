########################################################################################
##
##                          Testing brusselator system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54,
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestBrusselatorSystem(unittest.TestCase):
    """Brusselator system as an ODE block"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        # parameters
        self.a, self.b = 0.4, 1.2

        xy_0 = np.zeros(2)

        def f_bru(_x, u, t):
            x, y = _x
            dxdt = self.a - x - self.b * x + x**2 * y
            dydt = self.b * x - x**2 * y
            return np.array([dxdt, dydt])

        bru = ODE(func=f_bru, initial_value=xy_0)
        self.sco = Scope(labels=["x", "y"])

        blocks = [bru, self.sco]

        connections = [
            Connection(bru[:2], self.sco[:2])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )

        # build reference solution with scipy integrator
        def f_bru_(t, _x):
            x, y = _x
            dxdt = self.a - x - self.b * x + x**2 * y
            dydt = self.b * x - x**2 * y
            return np.array([dxdt, dydt])
        sol = solve_ivp(f_bru_, [0, 50], xy_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    time, [x, y] = self.sco.read()
                    xr, yr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
