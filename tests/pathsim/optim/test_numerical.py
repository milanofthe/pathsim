########################################################################################
##
##                           TESTS FOR 'optime/numerical.py'
##
##                                 Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.numerical import num_jac, num_autojac


class TestNumericalJacobian(unittest.TestCase):
    """
    testing of numerical jacobian calculation
    """

    def test_num_jac_scalar(self):

        #test scalar case
        def _f(x): return x**2
        def _df(x): return 2*x
        self.assertAlmostEqual(num_jac(_f, 3), _df(3), 4)
        self.assertAlmostEqual(num_jac(_f, -6.0), _df(-6.0), 4)
        self.assertAlmostEqual(num_jac(_f, 100), _df(100), 3)


    def test_num_jac_vec(self):

        #test vectorial case
        def _f(x): return np.array([np.cos(x[0]), np.sin(x[1])])
        def _df(x): return np.array([[-np.sin(x[0]), 0.0], [0.0, np.cos(x[1])]])
        self.assertAlmostEqual(np.sum(np.abs(num_jac(_f, np.ones(2)) - _df(np.ones(2)))), 0.0, 6)


    def test_num_autojac_scalar(self):

        #test scalar case
        def _f(x): return x**2
        def _df(x): return 2*x
        _j = num_autojac(_f)
        self.assertAlmostEqual(_j(3), _df(3), 6)
        self.assertAlmostEqual(_j(-6.0), _df(-6.0), 6)
        self.assertAlmostEqual(_j(100), _df(100), 3)


    def test_num_autojac_vec(self):

        #test vectorial case
        def _f(x): return np.array([np.cos(x[0]), np.sin(x[1])])
        def _df(x): return np.array([[-np.sin(x[0]), 0.0], [0.0, np.cos(x[1])]])
        _j = num_autojac(_f)
        self.assertAlmostEqual(np.sum(np.abs(_j(np.ones(2)) - _df(np.ones(2)))), 0.0, 6)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)