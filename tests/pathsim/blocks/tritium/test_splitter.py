########################################################################################
##
##                                  TESTS FOR 
##                         'blocks.tritium.splitter.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.tritium import Splitter

from tests.pathsim.blocks._embedding import Embedding



# TESTS ================================================================================

class TestFusionSplitter(unittest.TestCase):
    """
    Test the implementation of the 'Splitter' block class from the fusion toolbox
    """

    def test_init(self):
        
        #default initialization
        S = Splitter()
        self.assertEqual(S.fractions, np.ones(1))

        #input validation
        for fracs in [[1, 3], [0.4, 0.6, 0.001], [0.33, 0.33, 0.33]]:
            with self.assertRaises(ValueError):
                S = Splitter(fracs)

        #special initialization
        S = Splitter([0.4, 0.5, 0.1])
        self.assertEqual(sum(S.fractions - np.array([0.4, 0.5, 0.1])), 0)

        #test the automatic port maps
        self.assertEqual(S._port_map_out, {"out 0.4":0, "out 0.5":1, "out 0.1":2})


    def test_update(self):

        S = Splitter([0.4, 0.5, 0.1])

        #set block inputs
        S.inputs[0] = 2

        #update block
        S.update(None)

        #test if update was correct
        self.assertEqual(S.outputs[0], 0.8)
        self.assertEqual(S.outputs[1], 1)
        self.assertEqual(S.outputs[2], 0.2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
