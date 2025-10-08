########################################################################################
##
##                          Testing FitzHugh-Nagumo system
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
    Adder,
    Amplifier,
    Constant,
    Function
    )

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54,
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestFitzHughNagumoSystem(unittest.TestCase):
    """FitzHugh-Nagumo system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #parameters
        self.a, self.b, self.tau, self.R, self.I_ext = 0.7, 0.8, 12.5, 1.0, 0.5

        #system definition
        Iv = Integrator()
        Iw = Integrator()
        F3 = Function(lambda x: 1/3 * x**3)
        Ca = Constant(self.a)
        CR = Constant(self.R*self.I_ext)
        Gb = Amplifier(self.b)
        Gt = Amplifier(1/self.tau)
        Av = Adder("+--+")
        Aw = Adder("++-")

        self.Sc = Scope(labels=["v", "w"])

        blocks = [Iv, Iw, F3, Ca, CR, Gb, Gt, Av, Aw, self.Sc]

        #the connections between the blocks
        connections = [
            Connection(Av, Iv),
            Connection(Gt, Iw),
            Connection(Aw, Gt),
            Connection(Iv, Av[0], Aw[0], F3, self.Sc[0]),
            Connection(Iw, Av[2], Gb, self.Sc[1]),
            Connection(F3, Av[1]),
            Connection(CR, Av[3]),
            Connection(Ca, Aw[1]),
            Connection(Gb, Aw[2])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )

        # build reference solution with scipy integrator
        vw_0 = np.array([0.0, 0.0])
        def f_fhn(t, _x):
            v, w = _x
            dvdt = v - (1/3)*v**3 - w + self.R*self.I_ext
            dwdt = (v + self.a - self.b*w) / self.tau
            return np.array([dvdt, dwdt])
        sol = solve_ivp(f_fhn, [0, 50], vw_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    time, [v, w] = self.Sc.read()
                    vr, wr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(v-vr)), tol, 2)


class TestFitzHughNagumoODE(unittest.TestCase):
    """FitzHugh-Nagumo system as an ODE block"""

    def setUp(self):

        #parameters
        self.a, self.b, self.tau, self.R, self.I_ext = 0.7, 0.8, 12.5, 1.0, 0.5

        vw_0 = np.array([0.0, 0.0])

        def f_fhn(_x, u, t):
            v, w = _x
            dvdt = v - (1/3)*v**3 - w + self.R*self.I_ext
            dwdt = (v + self.a - self.b*w) / self.tau
            return np.array([dvdt, dwdt])

        fhn = ODE(f_fhn, initial_value=vw_0)

        self.sco = Scope(labels=["v", "w"])

        self.Sim = Simulation(
            blocks=[fhn, self.sco],
            connections=[
                Connection(fhn[0], self.sco[0]),
                Connection(fhn[1], self.sco[1])
                ],
            log=False
            )

        # build reference solution with scipy integrator
        def f_fhn_(t, _x):
            v, w = _x
            dvdt = v - (1/3)*v**3 - w + self.R*self.I_ext
            dwdt = (v + self.a - self.b*w) / self.tau
            return np.array([dvdt, dwdt])
        sol = solve_ivp(f_fhn_, [0, 50], vw_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
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

                    time, [v, w] = self.sco.read()
                    vr, wr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(v-vr)), tol, 2)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
