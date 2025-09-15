########################################################################################
##
##                                     TESTS FOR 
##                                  'blocks.ode.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.ode import ODE

#base solver for testing
from pathsim.solvers._solver import Solver 


# TESTS ================================================================================

class TestODE(unittest.TestCase):
    """
    Test the implementation of the 'ODE' block class
    """

    def test_init(self):

        #test default initialization
        D = ODE()

        self.assertEqual(D.engine, None)
        self.assertEqual(D.jac, None)
        self.assertEqual(D.initial_value, 0.0)

        #test special initialization
        def f(x, u, t): 
            return -x**2
        def J(x, u, t):
            return -2*x

        D = ODE(func=f, initial_value=1.0, jac=J)

        #test that ode function is correctly assigned
        self.assertEqual(D.func(1, 0, 0), f(1, 0, 0))
        self.assertEqual(D.func(2, 0, 0), f(2, 0, 0))
        self.assertEqual(D.func(3, 0, 0), f(3, 0, 0))

        #test that ode jacobian is correctly assigned
        self.assertEqual(D.jac(1, 0, 0), J(1, 0, 0))
        self.assertEqual(D.jac(2, 0, 0), J(2, 0, 0))
        self.assertEqual(D.jac(3, 0, 0), J(3, 0, 0))

        self.assertEqual(D.initial_value, 1.0)


    def test_len(self):

        D = ODE()

        #has direct passthrough
        self.assertEqual(len(D), 0)


    def test_set_solver(self):

        def f(x, u, t): 
            return -x**2
        def J(x, u, t):
            return -2*x

        D = ODE(func=f, initial_value=1.0, jac=J)

        #test that no solver is initialized
        self.assertEqual(D.engine, None)

        D.set_solver(Solver, None, tolerance_lte_rel=1e-4, tolerance_lte_abs=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(D.engine, Solver))
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-4)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-6)

        D.set_solver(Solver, None, tolerance_lte_rel=1e-3, tolerance_lte_abs=1e-4)

        #test that solver tolerance is changed
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-3)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-4)


    def test_operators(self):

        def f(x, u, t): 
            return -x**2
        def J(x, u, t):
            return -2*x

        D = ODE(func=f, initial_value=1.0, jac=J)

        self.assertEqual(D.op_alg, None)

        self.assertEqual(D.op_dyn(1, 2, 3), f(1, 2, 3))
        self.assertEqual(D.op_dyn(3, 2, 1), f(3, 2, 1))
        self.assertEqual(D.op_dyn(-2, 100, 3), f(-2, 100, 3))
        self.assertEqual(D.op_dyn(0.02, 0.1, 0), f(0.02, 0.1, 3))

        self.assertEqual(D.op_dyn.jac_x(1, 2, 3), J(1, 2, 3))
        self.assertEqual(D.op_dyn.jac_x(3, 2, 1), J(3, 2, 1))
        self.assertEqual(D.op_dyn.jac_x(-2, 100, 3), J(-2, 100, 3))
        self.assertEqual(D.op_dyn.jac_x(0.02, 0.1, 0), J(0.02, 0.1, 3))


    def test_update(self):

        D = ODE()
        D.set_solver(Solver, None)

        D.update(0)

        #test if engine state is retrieved correctly
        self.assertEqual(D.outputs[0], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
