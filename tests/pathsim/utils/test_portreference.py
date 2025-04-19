########################################################################################
##
##                                  TESTS FOR 
##                           'utils.portreference.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks._block import Block
from pathsim.utils.portreference import PortReference


# TESTS ================================================================================

class TestPortReference(unittest.TestCase):
    """
    test the 'PortReference' container class
    """

    def test_init(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(PR.block, B)
        self.assertEqual(PR.ports, [0])

        #special
        PR = PortReference(B, [0])
        self.assertEqual(PR.ports, [0]) #<- same as default

        PR = PortReference(B, [0, 1, 2, 3])
        self.assertEqual(PR.ports, [0, 1, 2, 3])

        PR = PortReference(B, [2, 3, 1, 3])
        self.assertEqual(PR.ports, [2, 3, 1, 3])









# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)