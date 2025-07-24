########################################################################################
##
##                                  TESTS FOR 
##                             'blocks.filters.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.filters import (
    ButterworthLowpassFilter,
    ButterworthHighpassFilter,
    ButterworthBandpassFilter,
    AllpassFilter
    )

# TESTS ================================================================================

class TestButterworthLowpassFilter(unittest.TestCase):
    """
    Test the implementation of the base 'ButterworthLowpassFilter' class
    """

    def test_init(self):

        flt = ButterworthLowpassFilter(Fc=100, n=2)

        self.assertEqual(flt.A.shape, (2, 2))


class TestButterworthHighpassFilter(unittest.TestCase):
    """
    Test the implementation of the base 'ButterworthHighpassFilter' class
    """

    def test_init(self):

        flt = ButterworthHighpassFilter(Fc=100, n=2)

        self.assertEqual(flt.A.shape, (2, 2))


class TestButterworthBandpassFilter(unittest.TestCase):
    """
    Test the implementation of the base 'ButterworthBandpassFilter' class
    """

    def test_init(self):

        flt = ButterworthBandpassFilter(Fc=[10, 100], n=4)
        
        self.assertEqual(flt.A.shape, (8, 8))

        with self.assertRaises(ValueError):
            flt = ButterworthBandpassFilter(Fc=[100], n=4)


class TestAllpassFilter(unittest.TestCase):
    """
    Test the implementation of the base 'AllpassFilter' class
    """

    def test_init(self):

        flt = AllpassFilter(fs=200)
        
        self.assertEqual(flt.A.shape, (1, 1))

        



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)