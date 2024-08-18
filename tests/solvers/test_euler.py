########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/euler.py'
##
##                            Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.euler import EUF, EUB


# TEST PROBLEM =========================================================================

def func(x, u, t):
    return -x

def jac(x, u, t):
    return -1

def solution(t):
    return np.exp(-t)


# TESTS ================================================================================

class EUFTest(unittest.TestCase):
    """
    Test the implementation of the Explicit Euler Forward (EUF) class
    """

    def setUp(self):
        self.solver = EUF(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertTrue(self.solver.is_explicit)
        self.assertFalse(self.solver.is_implicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_step(self):

        for i, t in enumerate(self.solver.stages(0, 1)):

            success, err, scale = self.solver.step(0.0, t, 1)

            #test if stage incrementation works
            self.assertEqual(self.solver.stage, i)

            #test if expected return
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 10)
        errors = []

        for dt in timesteps:
            self.solver.reset()
            time, numerical_solution = self.solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

            analytical_solution = solution(time)
            errors.append(np.linalg.norm(numerical_solution - analytical_solution))

        #test if errors are monotonically decreasing
        self.assertTrue(np.all(np.diff(errors)>0))

        #test convergence order, expected 1
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertEqual(np.round(p), 1)



class EUBTest(unittest.TestCase):
    """
    Test the implementation of the Implicit Euler Backward (EUB) class
    """

    def setUp(self):
        self.solver = EUB(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertFalse(self.solver.is_explicit)
        self.assertTrue(self.solver.is_implicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_step(self):

        for i, t in enumerate(self.solver.stages(0, 1)):

            success, err, scale = self.solver.step(0.0, t, 1)

            #test if stage incrementation works
            self.assertEqual(self.solver.stage, i)

            #test if expected return
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 10)
        errors = []

        for dt in timesteps:
            self.solver.reset()
            time, numerical_solution = self.solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

            analytical_solution = solution(time)
            errors.append(np.linalg.norm(numerical_solution - analytical_solution))

        #test if errors are monotonically decreasing
        self.assertTrue(np.all(np.diff(errors)>0))

        #test convergence order, expected 1
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertEqual(np.round(p), 1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)