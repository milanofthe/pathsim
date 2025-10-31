########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.multiplier.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.multiplier import Multiplier

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestMultiplier(unittest.TestCase):
    """
    Test the implementation of the 'Multiplier' block class
    """


    def test_embedding(self):

        M = Multiplier()

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) * np.sin(t) * 3.0 * t

        E = Embedding(M, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_MIMO(t))


        M = Multiplier()

        def src(t): return np.cos(t)
        def ref(t): return np.cos(t)

        E = Embedding(M, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_linearization(self):
        """test linearization and delinearization"""

        M = Multiplier()

        def src(t): return np.cos(t), t, 3.0
        def ref(t): return np.cos(t) * t * 3.0 

        E = Embedding(M, src, ref)

        for t in range(10): 
            self.assertEqual(*E.check_MIMO(t))

        #linearize block
        M.linearize(t)

        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        M.delinearize()

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_update_single(self):
        
        M = Multiplier()

        #set block inputs
        M.inputs[0] = 1

        #update block
        M.update(None)

        #test if update was correct
        self.assertEqual(M.outputs[0], 1)


    def test_update_multi(self):
        
        M = Multiplier()

        #set block inputs
        M.inputs[0] = 1
        M.inputs[1] = 2.0
        M.inputs[2] = 3.1

        #update block
        M.update(None)

        #test if update was correct
        self.assertEqual(M.outputs[0], 6.2)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)