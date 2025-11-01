########################################################################################
##
##                      Testing harmonic oscillator system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Scope

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87,
    ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85
    )


# TESTCASE =============================================================================

class TestHarmonicOscillatorSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #initial position and velocity
        self.x0, self.v0 = 2, 5

        #parameters (mass, damping, spring constant)
        self.m, self.c, self.k = 0.8, 0.2, 1.5

        #blocks that define the system
        I1 = Integrator(self.v0)   # integrator for velocity
        I2 = Integrator(self.x0)   # integrator for position
        A1 = Amplifier(self.c)
        A2 = Amplifier(self.k)
        A3 = Amplifier(-1/self.m)
        P1 = Adder()
        self.Sco = Scope(labels=["velocity", "position"])

        blocks = [I1, I2, A1, A2, A3, P1, self.Sco]

        #connections between the blocks
        connections = [
            Connection(I1, I2, A1, self.Sco),
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
            tolerance_fpi=1e-6,
            log=False,
            )


    def _reference(self, t):
        #analytical reference solution of damped harmonic oscillator
        omega0 = np.sqrt(self.k / self.m)
        zeta = self.c / (2 * np.sqrt(self.k * self.m))
        omega_d = omega0 * np.sqrt(1 - zeta**2)

        A = self.x0
        B = (self.v0 + zeta * omega0 * self.x0) / omega_d

        x = np.exp(-zeta * omega0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        v = np.exp(-zeta * omega0 * t) * (
            (-zeta * omega0 * A + omega_d * B) * np.cos(omega_d * t) +
            (-zeta * omega0 * B - omega_d * A) * np.sin(omega_d * t)
        )

        return v, x


    def test_eval_explicit_solvers(self):

        #test for different solvers
        for SOL in [RKBS32, RKF45, RKCK54, RKV65, RKDP87]:

            #test for different tolerances
            for tol in [1e-4, 1e-6, 1e-8]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL,
                        tolerance_lte_rel=0.0,
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(30)

                    time, [v, x] = self.Sco.read()
                    vr, xr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)


    def test_eval_implicit_solvers(self):

        #test for different solvers
        for SOL in [ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85]:

            #test for different tolerances
            for tol in [1e-4, 1e-6, 1e-8]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL,
                        tolerance_lte_rel=0.0,
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(30)

                    time, [v, x] = self.Sco.read()
                    vr, xr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
