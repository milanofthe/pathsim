########################################################################################
##
##                            Testing nonlinear roessler system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection
from pathsim.blocks import (
    ODE,
    Scope,
    Integrator,
    Constant,
    Adder,
    Amplifier,
    Multiplier
    )

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54,
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestRoesslerSystem(unittest.TestCase):
    """Roessler system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        # parameters a, b, c
        self.a, self.b, self.c = 0.2, 0.2, 5.7

        # Initial conditions
        x0, y0, z0 = 1.0, 1.0, 1.0

        # Integrators store the state variables x, y, z
        itg_x = Integrator(x0) # dx/dt = -y - z
        itg_y = Integrator(y0) # dy/dt = x + a*y
        itg_z = Integrator(z0) # dz/dt = b + z*(x - c)

        # Components for dx/dt
        add_neg_yz = Adder("--") # Computes -y - z

        # Components for dy/dt
        amp_a = Amplifier(self.a)     # Computes a*y
        add_x_ay = Adder("++")        # Computes x + (a*y)

        # Components for dz/dt
        cns_b = Constant(self.b)
        cns_c = Constant(self.c)
        add_x_c = Adder("+-")         # Computes x - c
        mul_z_xc = Multiplier()       # Computes z * (x - c)
        add_b_zxc = Adder("++")       # Computes b + [z * (x - c)]

        # Scope for plotting
        self.sco = Scope(labels=["x", "y", "z"])

        # List of all blocks
        blocks = [
            itg_x, itg_y, itg_z,
            add_neg_yz,
            amp_a, add_x_ay,
            cns_b, cns_c, add_x_c,
            mul_z_xc, add_b_zxc,
            self.sco
            ]

        # Connections
        connections = [
            # Output signals (from integrators)
            Connection(itg_x, add_x_ay[0], add_x_c[0], self.sco[0]),    # x connects to: (x + ay), (x - c), scope
            Connection(itg_y, add_neg_yz[0], amp_a, self.sco[1]),       # y connects to: (-y - z), a*y, scope
            Connection(itg_z, add_neg_yz[1], mul_z_xc[0], self.sco[2]), # z connects to: (-y - z), z*(x - c), scope

            # dx/dt path: -y - z -> itg_x
            Connection(add_neg_yz, itg_x),       # -y - z -> integrator x

            # dy/dt path: x + a*y -> itg_y
            Connection(amp_a, add_x_ay[1]),      # a*y -> x + (a*y) input 1
            Connection(add_x_ay, itg_y),         # x + a*y -> integrator y

            # dz/dt path: b + z*(x - c) -> itg_z
            Connection(cns_b, add_b_zxc[0]),     # b -> b + [...] input 0
            Connection(cns_c, add_x_c[1]),       # c -> x - c input 1
            Connection(add_x_c, mul_z_xc[1]),    # (x - c) -> z * (x - c) input 1
            Connection(mul_z_xc, add_b_zxc[1]),  # z * (x - c) -> b + [z * (x - c)] input 1
            Connection(add_b_zxc, itg_z)         # b + z*(x - c) -> integrator z
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )

        # build reference solution with scipy integrator
        xyz_0 = np.array([1.0, 1.0, 1.0])
        def f_roessler(t, _x):
            x, y, z = _x
            return np.array([-y - z, x + self.a*y, self.b + z*(x - self.c)])
        sol = solve_ivp(f_roessler, [0, 50], xyz_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    time, [x, y, z] = self.sco.read()
                    xr, yr, zr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)


class TestRoesslerODE(unittest.TestCase):
    """Roessler system as an ODE block"""

    def setUp(self):

        # parameters a, b, c
        self.a, self.b, self.c = 0.2, 0.2, 5.7

        xyz_0 = np.array([1.0, 1.0, 1.0])

        def f_roessler(_x, u, t):
            x, y, z = _x
            return np.array([-y - z, x + self.a*y, self.b + z*(x - self.c)])

        roe = ODE(f_roessler, initial_value=xyz_0)

        self.sco = Scope(labels=["x", "y", "z"])

        self.Sim = Simulation(
            blocks=[roe, self.sco],
            connections=[
                Connection(roe[0], self.sco[0]),
                Connection(roe[1], self.sco[1]),
                Connection(roe[2], self.sco[2])
                ],
            log=False
            )

        # build reference solution with scipy integrator
        def f_roessler_(t, _x):
            x, y, z = _x
            return np.array([-y - z, x + self.a*y, self.b + z*(x - self.c)])
        sol = solve_ivp(f_roessler_, [0, 50], xyz_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    time, [x, y, z] = self.sco.read()
                    xr, yr, zr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
