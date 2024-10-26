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

from ._referenceproblems import problems


# TESTS ================================================================================

class TestBDF2(unittest.TestCase):
    """
    Test the implementation of the 'BDF2' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF2()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = BDF2(initial_value=1, 
                        func=lambda x, u, t: -x, 
                        jac=lambda x, u, t: -1, 
                        tolerance_lte_rel=1e-3, 
                        tolerance_lte_abs=1e-6)

        self.assertEqual(solver.func(2, 0, 0), -2)
        self.assertEqual(solver.jac(2, 0, 0), -1)
        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = BDF2()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_buffer(self):

        solver = BDF2()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(solver.B)
            self.assertEqual(buffer_length, k+1 if k < 2 else 2)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err_rel, err_abs, scale = solver.step(0.0, t, 1)


    def test_step(self):

        solver = BDF2()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err_rel, err_abs, scale = solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err_rel, 0.0)
            self.assertEqual(err_abs, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 10)

        for problem in problems:

            solver = BDF2(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 1
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertGreater(p, 1)



class TestBDF3(unittest.TestCase):
    """
    Test the implementation of the 'BDF3' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF3()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = BDF3(initial_value=1, 
                        func=lambda x, u, t: -x, 
                        jac=lambda x, u, t: -1, 
                        tolerance_lte_rel=1e-3, 
                        tolerance_lte_abs=1e-6)

        self.assertEqual(solver.func(2, 0, 0), -2)
        self.assertEqual(solver.jac(2, 0, 0), -1)
        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = BDF3()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_buffer(self):

        solver = BDF3()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(solver.B)
            self.assertEqual(buffer_length, k+1 if k < 3 else 3)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err_rel, err_abs, scale = solver.step(0.0, t, 1)


    def test_step(self):

        solver = BDF3()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err_rel, err_abs, scale = solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err_rel, 0.0)
            self.assertEqual(err_abs, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 10)

        for problem in problems:

            solver = BDF3(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 1
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertGreater(p, 1)



class TestBDF4(unittest.TestCase):
    """
    Test the implementation of the 'BDF4' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF4()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = BDF4(initial_value=1, 
                        func=lambda x, u, t: -x, 
                        jac=lambda x, u, t: -1, 
                        tolerance_lte_rel=1e-3, 
                        tolerance_lte_abs=1e-6)

        self.assertEqual(solver.func(2, 0, 0), -2)
        self.assertEqual(solver.jac(2, 0, 0), -1)
        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = BDF4()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_buffer(self):

        solver = BDF4()

        #perform some steps
        for k in range(10):

            #test bdf buffer length
            buffer_length = len(solver.B)
            self.assertEqual(buffer_length, k+1 if k < 4 else 4)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err_rel, err_abs, scale = solver.step(0.0, t, 1)


    def test_step(self):

        solver = BDF4()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err_rel, err_abs, scale = solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err_rel, 0.0)
            self.assertEqual(err_abs, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-2, -1, 10)

        for problem in problems:

            solver = BDF4(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 1
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertGreater(p, 1)