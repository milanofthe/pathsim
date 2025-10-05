########################################################################################
##
##                          Testing stiff robertson system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ODE

from pathsim.solvers import (
    ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85,
    GEAR21, GEAR32, GEAR43, GEAR54, GEAR52A
    )


# TESTCASE =============================================================================

class TestRobertsonSystem(unittest.TestCase):
    """Robertson stiff system as an ODE block"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        # parameters
        self.a, self.b, self.c = 0.04, 1e4, 3e7

        # initial condition
        x0 = np.array([1, 0, 0])

        def func(x, u, t):
            return np.array([
                -self.a*x[0] + self.b*x[1]*x[2],
                 self.a*x[0] - self.b*x[1]*x[2] - self.c*x[1]**2,
                                                  self.c*x[1]**2
            ])

        # blocks that define the system
        Rob = ODE(func, x0)
        self.Sco = Scope(labels=["x", "y", "z"])

        blocks = [Rob, self.Sco]

        # the connections between the blocks
        connections = [
            Connection(Rob[0], self.Sco[0]),
            Connection(Rob[1], self.Sco[1]),
            Connection(Rob[2], self.Sco[2])
            ]

        # initialize simulation with the blocks, connections
        self.Sim = Simulation(
            blocks,
            connections,
            log=False,
            tolerance_fpi=1e-9
            )

        # build reference solution with scipy integrator
        def f_robertson(t, _x):
            return np.array([
                -self.a*_x[0] + self.b*_x[1]*_x[2],
                 self.a*_x[0] - self.b*_x[1]*_x[2] - self.c*_x[1]**2,
                                                     self.c*_x[1]**2
            ])
        sol = solve_ivp(f_robertson, [0, 10], x0, method="Radau", rtol=0.0, atol=1e-12, dense_output=True)
        self._reference = sol.sol


    def test_eval_implicit_solvers(self):

        #test for different solvers
        for SOL in [ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85, GEAR21, GEAR32, GEAR43, GEAR54, GEAR52A]:

            #test for different tolerances
            for tol in [1e-6]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL,
                        tolerance_lte_rel=0.0,
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(10)

                    time, [x, y, z] = self.Sco.read()
                    xr, yr, zr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.mean(abs(x-xr)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
