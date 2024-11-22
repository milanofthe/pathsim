########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/rkf78.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.rkf78 import RKF78

from ._referenceproblems import problems


# TESTS ================================================================================

class TestRKF78(unittest.TestCase):
    """
    Test the implementation of the 'RKF78' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = RKF78()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_explicit)
        self.assertFalse(solver.is_implicit)
        
        #test specific initialization
        solver = RKF78(initial_value=1, 
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

        solver = RKF78()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = RKF78()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err, scale = solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            if i < len(solver.eval_stages)-1:
                self.assertTrue(success)
                self.assertEqual(err, 0.0)
                self.assertEqual(scale, 1.0)

        #test if expected return at final stage
        self.assertNotEqual(err, 0.0)
        self.assertNotEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-1, 0, 20)

        for problem in problems:

            solver = RKF78(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=1.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 7 or 8
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertGreater(p, 6.5)


    def test_integrate_adaptive(self):

        #test the error control for each reference problem

        for problem in problems:

            solver = RKF78(problem.x0, problem.func, problem.jac, tolerance_lte_abs=1e-6, tolerance_lte_rel=1e-9)

            time, numerical_solution = solver.integrate(time_start=0.0, time_end=1.0, dt=1, adaptive=True)
            error = np.linalg.norm(numerical_solution - problem.solution(time))

            #test if error control was successful
            self.assertLess(error, solver.tolerance_lte_abs)