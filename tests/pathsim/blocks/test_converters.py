########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.converters.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.converters import DAC, ADC
from pathsim.events.schedule import Schedule


# TESTS ================================================================================

class TestADC(unittest.TestCase):
    """
    Test the implementation of the base 'ADC' class
    """

    def test_init(self):

        adc = ADC()

        self.assertTrue(isinstance(adc.events[0], Schedule))


    def test_len(self):

        adc = ADC()

        self.assertEqual(len(adc), 0)


class TestDAC(unittest.TestCase):
    """
    Test the implementation of the base 'DAC' class
    """

    def test_init(self):

        dac = DAC()

        self.assertTrue(isinstance(dac.events[0], Schedule))


    def test_len(self):

        dac = DAC()

        self.assertEqual(len(dac), 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)