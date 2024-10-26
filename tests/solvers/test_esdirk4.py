########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/esdirk4.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.esdirk4 import ESDIRK4

from ._referenceproblems import problems


# TESTS ================================================================================

class TestESDIRK4(unittest.TestCase):
    """
    Test the implementation of the 'ESDIRK4' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = ESDIRK4()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = ESDIRK4(initial_value=1, 
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

        solver = ESDIRK4()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = ESDIRK4()

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

        timesteps = np.logspace(-0.5, -0.1, 10)

        for problem in problems:

            solver = ESDIRK4(problem.x0, problem.func, problem.jac)
            
            errors = []

            for dt in timesteps:

                solver.reset()
                time, numerical_solution = solver.integrate(time_start=0.0, time_end=3.0, dt=dt, adaptive=False)

                errors.append(np.linalg.norm(numerical_solution - problem.solution(time)))

            #test if errors are monotonically decreasing
            self.assertTrue(np.all(np.diff(errors)>0))

            #test convergence order, expected 3 or 4
            p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
            self.assertEqual(round(p), 4)
