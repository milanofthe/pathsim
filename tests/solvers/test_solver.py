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
    ImplicitSolver)


# HELPER FUNCTIONS =====================================================================

def simple_func(x, u, t):
    return u - x

def simple_jac(x, u, t):
    return -1


# TESTS ================================================================================

class TestBaseSolver(unittest.TestCase):
    """
    Test the implementation of the base 'Solver' class
    """

    def setUp(self):
        self.solver = Solver(initial_value=1.0, func=simple_func, jac=simple_jac)

    def test_init(self):
        self.assertEqual(self.solver.x, 1.0)
        self.assertEqual(self.solver.x_0, 1.0)
        self.assertEqual(self.solver.initial_value, 1.0)
        self.assertEqual(self.solver.func, simple_func)
        self.assertEqual(self.solver.jac, simple_jac)
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
        self.solver.buffer()
        self.assertEqual(self.solver.x_0, 2.0)

    def test_change(self):
        new_solver = self.solver.change(ExplicitSolver)
        self.assertIsInstance(new_solver, ExplicitSolver)
        self.assertEqual(new_solver.get(), self.solver.get())


class ExplicitSolverTest(unittest.TestCase):
    """
    Test the implementation of the 'ExplicitSolver' base class
    """
    def setUp(self):
        self.solver = ExplicitSolver(initial_value=1.0, func=simple_func, jac=simple_jac)

    def test_init(self):
        self.assertTrue(self.solver.is_explicit)
        self.assertFalse(self.solver.is_implicit)

    def test_integrate_singlestep(self):
        success, err_rel, err_abs, scale = self.solver.integrate_singlestep(time=0, dt=0.1)
        self.assertTrue(success)
        self.assertEqual(err_rel, 0.0)
        self.assertEqual(err_abs, 0.0)
        self.assertEqual(scale, 1.0)

    def test_integrate(self):
        times, states = self.solver.integrate(time_start=0, time_end=1, dt=0.1)
        self.assertEqual(len(times), len(states))
        self.assertGreater(len(times), 1)
        self.assertEqual(times[0], 0)


class ImplicitSolverTest(unittest.TestCase):
    """
    Test the implementation of the 'ImplicitSolver' base class
    """

    def setUp(self):
        self.solver = ImplicitSolver(initial_value=1.0, func=simple_func, jac=simple_jac)

    def test_init(self):
        self.assertFalse(self.solver.is_explicit)
        self.assertTrue(self.solver.is_implicit)

    def test_solve(self):
        error = self.solver.solve(0, 0, 0.1)
        self.assertEqual(error, 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)