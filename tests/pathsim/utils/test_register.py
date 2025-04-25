########################################################################################
##
##                                  TESTS FOR 
##                             'utils.register.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.register import Register


# TESTS ================================================================================

class TestPortReference(unittest.TestCase):
    """
    test the 'Register' class
    """

    def test_init(self): pass
    def test_len(self): pass
    def test_iter(self): pass
    def test_reset(self): pass
    def test_to_array(self): pass
    def test_update_from_array(self): pass
    def test_update_from_array_max_err(self): pass
    def test_setitem(self): pass
    def test_getitem(self): pass


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)