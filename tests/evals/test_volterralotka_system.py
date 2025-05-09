########################################################################################
##
##                      Testing nonlinear volterra-lotka system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_ivp

from pathsim import Simulation, Connection

from pathsim.blocks import Scope, Integrator, Adder, Amplifier, Multiplier

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, 
    RKDP54, RKV65, RKF78, RKDP87
    )


# TESTCASE =============================================================================

class TestVolterraLotkaSystem(unittest.TestCase):
    """system built from distinct components"""

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        #parameters 
        alpha = 1.0  # growth rate of prey
        beta = 0.1   # predator sucess rate
        delta = 0.5  # predator efficiency
        gamma = 1.2  # death rate of predators

        #blocks that define the system
        i_pred = Integrator(10)
        i_prey = Integrator(5)

        a_alp = Amplifier(alpha)
        a_gma = Amplifier(gamma)
        a_bet = Amplifier(beta)
        a_del = Amplifier(delta)

        p_pred = Adder("-+")
        p_prey = Adder("+-")

        m_pp = Multiplier()

        self.sco = Scope(labels=["predator population", "prey population"])

        blocks = [
            i_pred, i_prey, a_alp, a_gma, 
            a_bet, a_del, p_pred, p_prey, 
            m_pp, self.sco
            ]

        #the connections between the blocks
        connections = [
            Connection(i_pred, m_pp[0], a_alp, self.sco[0]),
            Connection(i_prey, m_pp[1], a_gma, self.sco[1]),
            Connection(a_del, p_prey[0]),
            Connection(a_gma, p_prey[1]),
            Connection(a_bet, p_pred[0]),
            Connection(a_alp, p_pred[1]),
            Connection(m_pp, a_del, a_bet),
            Connection(p_pred, i_pred),
            Connection(p_prey, i_prey)
            ]


        #initialize the simulation with everything
        self.Sim = Simulation(
            blocks, 
            connections,
            log=False
            )
    
        #build reference solution with scipy integrator
        xy_0 = np.array([10, 5])
        def f_vl(t, _x):
            x, y = _x
            return np.array([alpha*x - beta*x*y, -gamma*y + delta*x*y])
        sol = solve_ivp(f_vl, [0, 20], xy_0, method="RK45", rtol=0.0, atol=1e-12, dense_output=True)
        self._reference = sol.sol


    def test_eval_solvers(self):

        #test for different solvers
        for SOL in [RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKDP87]:

            #test for different tolerances
            for tol in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):            

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL, 
                        tolerance_lte_rel=0.0, 
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(5)

                    time, [x, y] = self.sco.read()
                    xr, yr = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(x-xr)), tol, 2)  


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
