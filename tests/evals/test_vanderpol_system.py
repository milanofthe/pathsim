########################################################################################
##
##                                Testing vanderpol system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import Integrator, Scope, Adder, Multiplier, Amplifier, Function

from pathsim.solvers import (
    ESDIRK32, ESDIRK43, GEAR21, GEAR32, 
    GEAR43, GEAR54, GEAR52A
    )


# TESTCASE =============================================================================

class TestVanDerPolSystem(unittest.TestCase):
    """vanderpol system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        #initial condition
        x1_0 = 2
        x2_0 = 0

        #van der Pol parameter
        self.mu = 10 #(non stiff case)

        #blocks that define the system
        self.Sco = Scope()

        #subsystem with two separate integrators to emulate ODE block
        If = Interface()

        I1 = Integrator(x1_0)
        I2 = Integrator(x2_0)
        Fn = Function(lambda a: 1 - a**2)
        Pr = Multiplier()
        Ad = Adder("-+")
        Am = Amplifier(self.mu)

        sub_blocks = [If, I1, I2, Fn, Pr, Ad, Am]
        sub_connections = [
            Connection(I2, I1, Pr[0], If[1]), 
            Connection(I1, Fn, Ad[0], If[0]), 
            Connection(Fn, Pr[1]),
            Connection(Pr, Am),
            Connection(Am, Ad[1]),
            Connection(Ad, I2)
            ]

        #the subsystem acts just like a normal block
        VDP = Subsystem(sub_blocks, sub_connections)

        #blocks of the main system
        blocks = [VDP, self.Sco]

        #the connections between the blocks in the main system
        connections = [
            Connection(VDP, self.Sco)
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(
            blocks, 
            connections
            )

        #build reference solution with scipy integrator
        xy_0 = np.array([x1_0, x2_0])
        def f_vdp(t, x):
            return np.array([x[1], self.mu*(1 - x[0]**2)*x[1] - x[0]])
        sol = solve_ivp(f_vdp, [0, self.mu], xy_0, method="Radau", rtol=0.0, atol=1e-12, dense_output=True)
        self._reference = sol.sol


    def test_eval_solvers(self):

        #test for different solvers
        for SOL in [ESDIRK32, ESDIRK43, GEAR21, GEAR32, GEAR43, GEAR54, GEAR52A]:

            #test for different tolerances
            for tol in [1e-6]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):            

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL, 
                        tolerance_lte_rel=0.0, 
                        tolerance_lte_abs=tol,
                        tolerance_fpi=1e-12
                        )

                    self.Sim.run(self.mu)

                    time, [x] = self.Sco.read()
                    xr, yr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.mean(abs(x-xr)), tol, 2)  


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
