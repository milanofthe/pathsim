########################################################################################
##
##                      TESTS FOR 'solvers/_solver.py'
##
##                            Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np


from pathsim.solvers._solver import (
    Solver, 
    ExplicitSolver, 
    ImplicitSolver
    )


# TESTS ================================================================================

class TestBaseSolver(unittest.TestCase):
    """
    Test the implementation of the base 'Solver' class
    """

    def setUp(self):
        self.solver = Solver(initial_value=1.0)

    def test_init(self):
        self.assertEqual(self.solver.x, 1.0)
        self.assertEqual(self.solver.initial_value, 1.0)
        self.assertFalse(self.solver.is_adaptive)

    def test_str(self):
        self.assertEqual(str(self.solver), "Solver")

    def test_stages(self):
        stages = list(self.solver.stages(0, 1))
        self.assertEqual(stages, [0])

    def test_get_set(self):
        self.solver.set(2.0)
        self.assertEqual(self.solver.get(), 2.0)

    def test_reset(self):
        self.solver.set(2.0)
        self.solver.reset()
        self.assertEqual(self.solver.get(), 1.0)

    def test_buffer(self):
        self.solver.x = 2.0
        self.solver.buffer(0)
        self.assertEqual(self.solver.history[0], 2.0)

    def test_cast(self):
        new_solver = ExplicitSolver.cast(self.solver)
        self.assertIsInstance(new_solver, ExplicitSolver)
        self.assertEqual(new_solver.get(), self.solver.get())


class ExplicitSolverTest(unittest.TestCase):
    """
    Test the implementation of the 'ExplicitSolver' base class
    """
    def setUp(self):
        self.solver = ExplicitSolver(initial_value=1.0)

    def test_init(self):
        self.assertTrue(self.solver.is_explicit)
        self.assertFalse(self.solver.is_implicit)

    def test_integrate_singlestep(self):
        def func(x, t):
            return -x
        success, err, scale = self.solver.integrate_singlestep(func, time=0, dt=0.1)
        self.assertTrue(success)
        self.assertEqual(err, 0.0)
        self.assertEqual(scale, 1.0)

    def test_integrate(self):
        def func(x, t):
            return -x
        times, states = self.solver.integrate(func, time_start=0, time_end=1, dt=0.1)
        self.assertEqual(len(times), len(states))
        self.assertGreater(len(times), 1)
        self.assertEqual(times[0], 0)


class ImplicitSolverTest(unittest.TestCase):
    """
    Test the implementation of the 'ImplicitSolver' base class
    """

    def setUp(self):
        self.solver = ImplicitSolver(initial_value=1.0)

    def test_init(self):
        self.assertFalse(self.solver.is_explicit)
        self.assertTrue(self.solver.is_implicit)

    def test_solve(self):
        error = self.solver.solve(0, 0, 0.1)
        self.assertEqual(error, 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)