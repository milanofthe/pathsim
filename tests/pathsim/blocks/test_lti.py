########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.lti.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.lti import StateSpace, TransferFunction

#base solver for testing
from pathsim.solvers._solver import Solver 

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestStateSpace(unittest.TestCase):
    """
    Test the implementation of the 'StateSpace' block class
    """


    def test_init(self):

        #test default initialization
        S = StateSpace()
        self.assertEqual(S.initial_value, 0.0)
        self.assertEqual(S.engine, None)

        self.assertEqual(len(S.inputs), 1)
        self.assertEqual(len(S.outputs), 1)

        #test special initialization (siso)
        S = StateSpace(
            A=np.eye(2), 
            B=np.ones(2), 
            C=np.ones(2), 
            D=1, 
            initial_value=None
            )
        self.assertTrue(np.all(S.initial_value == np.zeros(2)))
        self.assertEqual(len(S.inputs), 1)
        self.assertEqual(len(S.outputs), 1)

        #test special initialization (mimo)
        S = StateSpace(
            A=np.eye(2), 
            B=np.ones((2, 2)), 
            C=np.ones((2, 2)), 
            D=np.ones((2, 2)), 
            initial_value=np.ones(2)
            )
        self.assertTrue(np.all(S.initial_value == np.ones(2)))

        #are the dimensions initialized correctly?
        self.assertEqual(len(S.inputs), 2)
        self.assertEqual(len(S.outputs), 2)


    def test_len(self):
        
        #no direct passthrough (siso)
        S = StateSpace(D=0)
        self.assertEqual(len(S), 0)

        #no direct passthrough (mimo)
        S = StateSpace(D=np.zeros(2))
        self.assertEqual(len(S), 0)

        #direct passthrough (siso)
        S = StateSpace(D=3)
        self.assertEqual(len(S), 1)

        #direct passthrough (mimo)
        S = StateSpace(D=np.array([0, 5]))
        self.assertEqual(len(S), 1)


    def test_set_solver(self):

        S = StateSpace(initial_value=1.0)

        #test that no solver is initialized
        self.assertEqual(S.engine, None)

        S.set_solver(Solver, tolerance_lte_rel=1e-4, tolerance_lte_abs=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(S.engine, Solver))

        #test that solver parametes have been set
        self.assertEqual(S.engine.tolerance_lte_rel, 1e-4)
        self.assertEqual(S.engine.tolerance_lte_abs, 1e-6)
        self.assertEqual(S.engine.initial_value, 1.0)

        S.set_solver(Solver, tolerance_lte_rel=1e-2, tolerance_lte_abs=1e-3)

        #test that solver tolerance is changed
        self.assertEqual(S.engine.tolerance_lte_rel, 1e-2)
        self.assertEqual(S.engine.tolerance_lte_abs, 1e-3)


    def test_update(self):

        S = StateSpace(initial_value=1.1)
        S.set_solver(Solver)

        #test if output is zero 
        self.assertEqual(S.get(0), 0.0)
        
        S.set(0, 3.3)
        err = S.update(0)

        #test if error is correctly 0
        self.assertGreater(err, 0.0)

        #test if engine state is calculated correctly
        self.assertAlmostEqual(S.get(0), 2.2, 8)


    def test_embedding_siso(self):

        S = StateSpace(
            A=np.eye(2), 
            B=np.ones(2), 
            C=np.ones(2), 
            D=0.5, 
            initial_value=None
            )    
        S.set_solver(Solver)
        
        def src(t): return np.sin(t)
        def ref(t): return 0.5*np.sin(t)

        E = Embedding(S, src, ref)
        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_embedding_mimo(self):

        S = StateSpace(
            A=np.eye(2), 
            B=np.ones((2, 2)), 
            C=np.ones((2, 2)), 
            D=np.ones((2, 2)), 
            initial_value=None
            )    
        S.set_solver(Solver)
        
        def src(t): return [np.sin(t), np.cos(t)]
        def ref(t): return np.array([np.sin(t) + np.cos(t), np.sin(t) + np.cos(t)])

        E = Embedding(S, src, ref)
        for t in range(10): 
            s, r = E.check_MIMO(t)
            self.assertTrue(np.all(s==r))


class TestTransferFunction(unittest.TestCase):
    """
    Test the implementation of the 'TransferFunction' block class

    inherits most methods from the 'StateSpace' block, so only 
    testing ot the initialization is required
    """

    def test_init(self):

        #test default initialization        
        with self.assertRaises(ValueError):
            T = TransferFunction()

        #test specific initialization (siso)
        T = TransferFunction(Poles=2, Residues=0.5, Const=5.5)
        self.assertEqual(T.A, 2)
        self.assertEqual(T.B, 1)
        self.assertEqual(T.C, 0.5)
        self.assertEqual(T.D, 5.5)

        #test specific initialization (mimo)
        T = TransferFunction(Poles=np.array([1, 2]), 
                             Residues=2*np.ones((2, 2)), 
                             Const=np.ones(2))
        self.assertTrue(np.all(T.A == np.diag(np.array([1, 2]))))
        self.assertTrue(np.all(T.B == np.ones(2)))
        self.assertTrue(np.all(T.C == 2*np.ones(2)))
        self.assertTrue(np.all(T.D == np.ones(2)))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
