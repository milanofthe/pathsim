########################################################################################
##
##                          Testing PID controller system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope, PID

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87,
    ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85
    )


# TESTCASE =============================================================================

class TestPIDSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #plant gain
        self.K = 0.4

        #pid parameters
        self.Kp, self.Ki, self.Kd = 1.5, 0.5, 0.1

        #source function
        def f_s(t):
            if t>6: return 0.5
            elif t>2: return 1
            else: return 0

        #blocks
        spt = Source(f_s)
        err = Adder("+-")
        pid = PID(self.Kp, self.Ki, self.Kd, f_max=10)
        pnt = Integrator()
        pgn = Amplifier(self.K)
        self.sco = Scope(labels=["s(t)", "x(t)", "error"])

        blocks = [spt, err, pid, pnt, pgn, self.sco]

        connections = [
            Connection(spt, err, self.sco[0]),
            Connection(pgn, err[1], self.sco[1]),
            Connection(err, pid, self.sco[2]),
            Connection(pid, pnt),
            Connection(pnt, pgn)
        ]

        #simulation initialization
        self.Sim = Simulation(
            blocks,
            connections,
            log=False
            )


    def _reference_response(self):
        #simple expected behavior: system should settle close to setpoint
        #we're testing for proper settling within tolerance
        return 1.0  # expected steady state for setpoint = 1


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

                    self.Sim.run(20)

                    time, [sp, ot, er] = self.sco.read()

                    #check that system settles near setpoint
                    #after initial transient (t>15), error should be small
                    final_error = np.abs(er[time > 15])
                    self.assertTrue(np.max(final_error) < 0.1)


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

                    self.Sim.run(20)

                    time, [sp, ot, er] = self.sco.read()

                    #check that system settles near setpoint
                    #after initial transient (t>15), error should be small
                    final_error = np.abs(er[time > 15])
                    self.assertTrue(np.max(final_error) < 0.1)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
