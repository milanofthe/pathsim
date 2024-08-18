########################################################################################
##
##                                  TESTS FOR 
##                               'solvers/bdf.py'
##
##                            Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.bdf import BDF2, BDF3, BDF4


# TEST PROBLEM =========================================================================

def func(x, u, t):
    return -x

def jac(x, u, t):
    return -1

def solution(t):
    return np.exp(-t)


# TESTS ================================================================================

class BDF2Test(unittest.TestCase):
    """
    Test the implementation of the 'BDF2' solver class
    """

    def setUp(self):
        self.solver = BDF2(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertTrue(self.solver.is_implicit)
        self.assertFalse(self.solver.is_explicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_buffer(self):

        self.solver.reset()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(self.solver.B)
            self.assertEqual(buffer_length, k+1 if k < 2 else 2)
            
            #make one step
            for i, t in enumerate(self.solver.stages(0, 1)):
                success, err, scale = self.solver.step(0.0, t, 1)


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

        #test convergence order, expected between 1 and 2 due to ramp up
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertGreater(p, 1.2)


class BDF3Test(unittest.TestCase):
    """
    Test the implementation of the 'BDF3' solver class
    """

    def setUp(self):
        self.solver = BDF3(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertTrue(self.solver.is_implicit)
        self.assertFalse(self.solver.is_explicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_buffer(self):

        self.solver.reset()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(self.solver.B)
            self.assertEqual(buffer_length, k+1 if k < 3 else 3)
            
            #make one step
            for i, t in enumerate(self.solver.stages(0, 1)):
                success, err, scale = self.solver.step(0.0, t, 1)


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

        #test convergence order, expected between 1 and 3 due to ramp up
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertGreater(p, 1.2)


class BDF4Test(unittest.TestCase):
    """
    Test the implementation of the 'BDF4' solver class
    """

    def setUp(self):
        self.solver = BDF4(initial_value=1, func=func, jac=jac)


    def test_init(self):
        self.assertTrue(self.solver.is_implicit)
        self.assertFalse(self.solver.is_explicit)


    def test_stages(self):
        for i, t in enumerate(self.solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, self.solver.eval_stages[i])


    def test_buffer(self):

        self.solver.reset()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(self.solver.B)
            self.assertEqual(buffer_length, k+1 if k < 4 else 4)
            
            #make one step
            for i, t in enumerate(self.solver.stages(0, 1)):
                success, err, scale = self.solver.step(0.0, t, 1)


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

        #test convergence order, expected between 1 and 4 due to ramp up
        p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
        self.assertGreater(p, 1.2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)