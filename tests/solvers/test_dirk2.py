########################################################################################
##
##                                  TESTS FOR 
##                              'solvers/dirk2.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.dirk2 import DIRK2

from ._referenceproblems import problems


# TESTS ================================================================================

class TestDIRK2(unittest.TestCase):
    """
    Test the implementation of the 'DIRK2' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = DIRK2()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = DIRK2(initial_value=1, 
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

        solver = DIRK2()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = DIRK2()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            _ = solver.solve(0.0, t, 1) #needed for implicit solvers to get slope
            success, err_rel, err_abs, scale = solver.step(0.0, t, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err_rel, 0.0)
            self.assertEqual(err_abs, 0.0)
            self.assertEqual(scale, 1.0)


    def test_integrate_fixed(self):
        
        #integrate test problem and assess convergence order

        timesteps = np.logspace(-1, -0.2, 10)

        for problem in problems:

            solver = DIRK2(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=3.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 2 or 3
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertEqual(round(p), 2)