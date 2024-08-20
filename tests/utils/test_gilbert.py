########################################################################################
##
##                                    TESTS FOR
##                               'utils/gilbert.py'
##
##                                Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
from pathsim.utils.gilbert import gilbert_realization


# HELPER FUNCTIONS =====================================================================

def evaluate_statespace(s, A, B, C, D):
    n = A.shape[0]
    return np.dot(C, np.linalg.solve(s*np.eye(n) - A, B)) + D

def evaluate_poleresidue(s, Poles, Residues, Const):
    return np.sum([r/(s - p) for p, r in zip(Poles, Residues)], axis=0) + Const


# TESTS ================================================================================

class TestGilbertRealization(unittest.TestCase):
    """
    Test the implementation of the gilbert statespace realization
    """

    def assertArrayAlmostEqual(self, first, second, places=7, msg=None):
        np.testing.assert_array_almost_equal(first, second, decimal=places, err_msg=msg)

    def test_helpers(self):
        A, B, C, D = np.array([[1]]), np.array([[1]]), np.array([[1]]), np.array([[1]])
        self.assertArrayAlmostEqual(evaluate_statespace(0, A, B, C, D), np.array([[0]]))
        self.assertArrayAlmostEqual(evaluate_statespace(1j, A, B, C, D), np.array([[1+1/(1j-1)]]))

        Poles, Residues, Const = [1], [np.array([1])], np.array([[1]])
        self.assertArrayAlmostEqual(evaluate_poleresidue(0, Poles, Residues, Const), np.array([[0]]))
        self.assertArrayAlmostEqual(evaluate_poleresidue(1j, Poles, Residues, Const), np.array([[1+1/(1j-1)]]))

    def test_siso_real_poles(self):
        Poles = [-1.0, -2.0, -3.0]
        Residues = [1.0, 2.0, 3.0]
        Const = 0.5
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        for s in [0, 1j, 10j, 100j]:
            ss_eval = evaluate_statespace(s, A, B, C, D)
            pr_eval = evaluate_poleresidue(s, Poles, Residues, Const)
            self.assertArrayAlmostEqual(ss_eval, pr_eval, places=12)

    def test_siso_complex_poles(self):
        Poles = [-1+1j, -1-1j, -2]
        Residues = [1-0.5j, 1+0.5j, 2]
        Const = 0.1
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        for s in [0, 1j, 10j, 100j]:
            ss_eval = evaluate_statespace(s, A, B, C, D)
            pr_eval = evaluate_poleresidue(s, Poles, Residues, Const)
            self.assertArrayAlmostEqual(ss_eval, pr_eval, places=12)

    def test_mimo_2x2(self):
        Poles = [-1, -2, -3]
        Residues = [np.array([[1, 2], [3, 4]]), 
                    np.array([[2, 3], [4, 5]]), 
                    np.array([[3, 4], [5, 6]])]
        Const = np.array([[0.1, 0.2], [0.3, 0.4]])
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        for s in [0, 1j, 10j, 100j]:
            ss_eval = evaluate_statespace(s, A, B, C, D)
            pr_eval = evaluate_poleresidue(s, Poles, Residues, Const)
            self.assertArrayAlmostEqual(ss_eval, pr_eval, places=12)

    def test_mimo_3x2(self):
        Poles = [-1+1j, -1-1j, -2]
        Residues = [np.array([[1, 2], [3, 4], [5, 6]]), 
                    np.array([[1, 2], [3, 4], [5, 6]]), 
                    np.array([[2, 3], [4, 5], [6, 7]])]
        Const = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        for s in [0, 1j, 10j, 100j]:
            ss_eval = evaluate_statespace(s, A, B, C, D)
            pr_eval = evaluate_poleresidue(s, Poles, Residues, Const)
            self.assertArrayAlmostEqual(ss_eval, pr_eval, places=12)

    def test_input_validation(self):

        # Empty poles and residues
        with self.assertRaises(ValueError):
            gilbert_realization([], [], 0)  
        
        # Mismatched poles and residues
        with self.assertRaises(ValueError):
            gilbert_realization([1, 2], [np.array([1])], 0)  


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)