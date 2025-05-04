########################################################################################
##
##                            Testing nonlinear lorenz system
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

class TestLorenzSystem(unittest.TestCase):
    """Lorenz system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        # parameters 
        sigma, rho, beta = 10, 28, 8/3

        # Initial conditions
        x0, y0, z0 = 1.0, 1.0, 1.0

        # Integrators store the state variables x, y, z
        itg_x = Integrator(x0) # dx/dt = sigma * (y - x)
        itg_y = Integrator(y0) # dy/dt = x * (rho - z) - y
        itg_z = Integrator(z0) # dz/dt = x * y - beta * z

        # Components for dx/dt
        amp_sigma = Amplifier(sigma)
        add_x = Adder("+-") # Computes y - x

        # Components for dy/dt
        cns_rho = Constant(rho)
        add_rho_z = Adder("+-") # Computes rho - z
        mul_x_rho_z = Multiplier() # Computes x * (rho - z)
        add_y = Adder("-+") # Computes -y + [x * (rho - z)]

        # Components for dz/dt
        mul_xy = Multiplier() # Computes x * y
        amp_beta = Amplifier(beta) # Computes beta * z
        add_z = Adder("+-") # Computes (x * y) - (beta * z)

        # Scope for plotting
        self.sco = Scope(labels=["x", "y", "z"])

        # List of all blocks
        blocks = [
            itg_x, itg_y, itg_z,
            amp_sigma, add_x,
            cns_rho, add_rho_z, mul_x_rho_z, add_y,
            mul_xy, amp_beta, add_z,
            self.sco
            ]

        # Connections
        connections = [
            # Output signals (from integrators)
            Connection(itg_x, add_x[1], mul_x_rho_z[0], mul_xy[0], self.sco[0]), # x -> (y-x), x*(rho-z), x*y, scope
            Connection(itg_y, add_x[0], add_y[0], mul_xy[1], self.sco[1]),       # y -> (y-x), -y + [...], x*y, scope
            Connection(itg_z, add_rho_z[1], amp_beta, self.sco[2]),              # z -> (rho-z), beta*z, scope

            # dx/dt path: sigma * (y - x) -> itg_x
            Connection(add_x, amp_sigma),  # (y - x) -> sigma * (y - x)
            Connection(amp_sigma, itg_x),  # sigma * (y - x) -> integrator x

            # dy/dt path: x * (rho - z) - y -> itg_y
            Connection(cns_rho, add_rho_z[0]),     # rho -> (rho - z) input 0
            Connection(add_rho_z, mul_x_rho_z[1]), # (rho - z) -> x * (rho - z) input 1
            Connection(mul_x_rho_z, add_y[1]),     # x * (rho - z) -> -y + [x * (rho - z)] input 1
            Connection(add_y, itg_y),              # x * (rho - z) - y -> integrator y

            # dz/dt path: x * y - beta * z -> itg_z
            Connection(mul_xy, add_z[0]),    # x * y -> (x * y) - (beta * z) input 0
            Connection(amp_beta, add_z[1]),  # beta * z -> (x * y) - (beta * z) input 1
            Connection(add_z, itg_z)         # (x * y) - (beta * z) -> integrator z
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )
    
        # build reference solution with scipy integrator
        xyz_0 = np.array([1.0, 1.0, 1.0])
        def f_lorenz(t, _x):
            x, y, z = _x
            return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        sol = solve_ivp(f_lorenz, [0, 5], xyz_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    self.Sim.run(5)

                    time, [x, y, z] = self.sco.read()
                    xr, yr, zr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)  


class TestLorenzODE(unittest.TestCase):
    """Lorenz system as an ODE block"""

    def setUp(self):

        # parameters 
        sigma, rho, beta = 10, 28, 8/3

        xyz_0 = np.array([1.0, 1.0, 1.0])

        def f_lorenz(_x, u, t):
            x, y, z = _x
            return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

        lor = ODE(f_lorenz, initial_value=xyz_0)

        self.sco = Scope(labels=["x", "y", "z"])

        self.Sim = Simulation(
            blocks=[lor, self.sco],
            connections=[
                Connection(lor[0], self.sco[0]),
                Connection(lor[1], self.sco[1]),
                Connection(lor[2], self.sco[2])
                ],
            log=False
            )

        # build reference solution with scipy integrator
        def f_lorenz_(t, _x):
            x, y, z = _x
            return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        sol = solve_ivp(f_lorenz_, [0, 5], xyz_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    self.Sim.run(5)

                    time, [x, y, z] = self.sco.read()
                    xr, yr, zr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)  




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
