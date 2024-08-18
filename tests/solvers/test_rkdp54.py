########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/rkdp54.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.rkdp54 import RKDP54


# TEST PROBLEM =========================================================================

def func(x, u, t):
    return -x

def jac(x, u, t):
    return -1

def solution(t):
    return np.exp(-t)


# TESTS ================================================================================

class RKDP54Test(unittest.TestCase):
    """
    Test the implementation of the 'RKDP54' solver class
    """

    def setUp(self):
        self.solver = RKDP54(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertTrue(self.solver.is_explicit)
        self.assertFalse(self.solver.is_implicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_step(self):

        for i, t in enumerate(self.solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(self.solver.stage, i)

            success, err, scale = self.solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            if i < len(self.solver.eval_stages)-1:
                self.assertTrue(success)
                self.assertEqual(err, 0.0)
                self.assertEqual(scale, 1.0)

        #test if expected return at final stage
        self.assertNotEqual(err, 0.0)
        self.assertNotEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 20)
        errors = []

        for dt in timesteps:
            self.solver.reset()
            time, numerical_solution = self.solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

            analytical_solution = solution(time)
            errors.append(np.linalg.norm(numerical_solution - analytical_solution))

        #test if errors are monotonically decreasing
        self.assertTrue(np.all(np.diff(errors)>0))

        #test convergence order, expected 3
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertEqual(np.round(p), 5)


    def test_integrate_adaptive(self):

        #test the error control

        self.solver.reset()
        time, numerical_solution = self.solver.integrate(time_start=0.0, time_end=1.0, dt=1, adaptive=True)
        error = np.linalg.norm(numerical_solution - solution(time))

        #test if error control was successful
        self.assertLess(error, self.solver.tolerance_lte)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)