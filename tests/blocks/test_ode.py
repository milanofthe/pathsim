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
        def j(x, u, t):
            return -2*x

        D = ODE(func=f, initial_value=1.0, jac=j)

        #test that ode function is correctly assigned
        self.assertEqual(D.func(1, 0, 0), f(1, 0, 0))
        self.assertEqual(D.func(2, 0, 0), f(2, 0, 0))
        self.assertEqual(D.func(3, 0, 0), f(3, 0, 0))

        #test that ode jacobian is correctly assigned
        self.assertEqual(D.jac(1, 0, 0), j(1, 0, 0))
        self.assertEqual(D.jac(2, 0, 0), j(2, 0, 0))
        self.assertEqual(D.jac(3, 0, 0), j(3, 0, 0))

        self.assertEqual(D.initial_value, 1.0)


    def test_len(self):

        D = ODE()

        #has direct passthrough
        self.assertEqual(len(D), 0)


    def test_str(self):

        D = ODE()

        #test default str method
        self.assertEqual(str(D), "ODE")


    def test_set_solver(self):

        def f(x, u, t): 
            return -x**2
        def j(x, u, t):
            return -2*x

        D = ODE(func=f, initial_value=1.0, jac=j)

        #test that no solver is initialized
        self.assertEqual(D.engine, None)

        D.set_solver(Solver, tolerance_lte=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(D.engine, Solver))
        self.assertEqual(D.engine.tolerance_lte, 1e-6)

        #test that solver function is correctly assigned
        self.assertEqual(D.engine.func(1, 0, 0), f(1, 0, 0))
        self.assertEqual(D.engine.func(2, 0, 0), f(2, 0, 0))
        self.assertEqual(D.engine.func(3, 0, 0), f(3, 0, 0))

        #test that solver jacobian is correctly assigned
        self.assertEqual(D.engine.jac(1, 0, 0), j(1, 0, 0))
        self.assertEqual(D.engine.jac(2, 0, 0), j(2, 0, 0))
        self.assertEqual(D.engine.jac(3, 0, 0), j(3, 0, 0))

        D.set_solver(Solver, tolerance_lte=1e-3)

        #test that solver tolerance is changed
        self.assertEqual(D.engine.tolerance_lte, 1e-3)


    def test_update(self):

        D = ODE()
        D.set_solver(Solver)

        err = D.update(0)

        #test if error is correctly 0
        self.assertEqual(err, 0.0)

        #test if engine state is retrieved correctly
        self.assertEqual(D.get(0), 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
