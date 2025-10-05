########################################################################################
##
##                          Testing nonlinear pendulum system
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
    Function,
    Adder,
    Scope
    )

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54,
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestPendulumSystem(unittest.TestCase):
    """Pendulum system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #initial angle and angular velocity
        phi0, omega0 = 0.9*np.pi, 0

        #parameters (gravity, length)
        self.g, self.l = 9.81, 1

        #blocks that define the system
        In1 = Integrator(omega0)
        In2 = Integrator(phi0)
        Amp = Amplifier(-self.g/self.l)
        Fnc = Function(np.sin)
        self.Sco = Scope(labels=["angular velocity", "angle"])

        blocks = [In1, In2, Amp, Fnc, self.Sco]

        #connections between the blocks
        connections = [
            Connection(In1, In2, self.Sco[0]),
            Connection(In2, Fnc, self.Sco[1]),
            Connection(Fnc, Amp),
            Connection(Amp, In1)
            ]

        #initialize simulation
        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )

        # build reference solution with scipy integrator
        phi_omega_0 = np.array([omega0, phi0])
        def f_pendulum(t, _x):
            omega, phi = _x
            return np.array([-(self.g/self.l)*np.sin(phi), omega])
        sol = solve_ivp(f_pendulum, [0, 15], phi_omega_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    self.Sim.run(15)

                    time, [omega, phi] = self.Sco.read()
                    omegar, phir = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(phi-phir)), tol, 2)


class TestPendulumODE(unittest.TestCase):
    """Pendulum system as an ODE block"""

    def setUp(self):

        #initial angle and angular velocity
        omega0, phi0 = 0, 0.9*np.pi

        #parameters (gravity, length)
        self.g, self.l = 9.81, 1

        phi_omega_0 = np.array([omega0, phi0])

        def f_pendulum(_x, u, t):
            omega, phi = _x
            return np.array([-(self.g/self.l)*np.sin(phi), omega])

        pen = ODE(f_pendulum, initial_value=phi_omega_0)

        self.sco = Scope(labels=["angular velocity", "angle"])

        self.Sim = Simulation(
            blocks=[pen, self.sco],
            connections=[
                Connection(pen[0], self.sco[0]),
                Connection(pen[1], self.sco[1])
                ],
            log=False
            )

        # build reference solution with scipy integrator
        def f_pendulum_(t, _x):
            omega, phi = _x
            return np.array([-(self.g/self.l)*np.sin(phi), omega])
        sol = solve_ivp(f_pendulum_, [0, 15], phi_omega_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    self.Sim.run(15)

                    time, [omega, phi] = self.sco.read()
                    omegar, phir = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(phi-phir)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
