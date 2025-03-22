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


    def test_update_single(self):
        
        M = Multiplier()

        #set block inputs
        M.set(0, 1)

        #update block
        err = M.update(None)

        #test if update was correct
        self.assertEqual(M.get(0), 1)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = M.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


    def test_embedding(self):

        M = Multiplier()

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) * np.sin(t) * 3.0 * t

        E = Embedding(M, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_update_multi(self):
        
        M = Multiplier()

        #set block inputs
        M.set(0, 1)
        M.set(1, 2.0)
        M.set(2, 3.1)

        #update block
        err = M.update(None)

        #test if update was correct
        self.assertEqual(M.get(0), 6.2)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = M.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)





# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)