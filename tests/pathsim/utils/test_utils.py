########################################################################################
##
##                              TESTS FOR 'utils/utils.py'
##
##                                Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.utils import (
    dict_to_array, 
    array_to_dict, 
    abs_error,
    rel_error,
    max_error,
    max_error_dicts,
    max_rel_error, 
    max_rel_error_dicts
    )


# TESTS ================================================================================

class TestUtilsFuncs(unittest.TestCase):
    """
    test all the array-dict conversions and functions for error calculation
    """

    def test_abs_error(self):

        self.assertEqual(abs_error(1.0, 2.0), 1.0)    
        self.assertEqual(abs_error(2.0, 1.0), 1.0)
        self.assertEqual(abs_error(2.0, 0.0), 2.0)
        self.assertEqual(abs_error(0.0, 1.0), 1.0)


    def test_rel_error(self):

        #test fallback to abs error
        self.assertEqual(rel_error(0.0, 1), 1)
        self.assertEqual(rel_error(0.0, 0.1), 0.1)

        #test a<b
        self.assertEqual(rel_error(1.0, 2.0), 1.0)
        
        #test a>b
        self.assertEqual(rel_error(2.0, 1.0), 0.5)


    def test_max_error(self):
        self.assertEqual(max_error([0.00139, 2.4, 87, 1, 97.8, 4.35], 
                                   [1.00139, 1.4, 86, 2, 98.2, 33.35]), 
                         29)


    def test_max_rel_error(self):
        self.assertEqual(max_rel_error([0.00139, 2.4, 87, 1, 97.8, 4.35], 
                                       [1.00139, 1.4, 86, 2, 98.2, 33.35]), 
                         1/0.00139)


    def test_dict_to_array(self):

        #test conversion
        self.assertEqual(np.sum(np.abs(dict_to_array({0:12, 1:3.2, 2:31.0})-np.array([12, 3.2, 31.0]))), 0.0)

        #test key sorting
        self.assertEqual(np.sum(np.abs(dict_to_array({0:12, 2:3.2, 1:31.0}) - np.array([12, 31.0, 3.2]))), 0.0)
        self.assertEqual(np.sum(np.abs(dict_to_array({1:12, 2:3.2, 3:31.0})- np.array([12, 3.2, 31.0]))), 0.0)

        #test non uniform keys
        self.assertEqual(np.sum(np.abs(dict_to_array({0:12, 2:3.2, 3:31.0}) - np.array([12, 3.2, 31.0]))), 0.0)


    def test_array_to_dict(self):

        #test scalar input
        self.assertEqual(array_to_dict(4), {0:4})

        #test array input
        self.assertEqual(array_to_dict(np.array([12, 3.2, 31.0])), {0:12, 1:3.2, 2:31.0})
        self.assertEqual(array_to_dict(np.array([2.0])), {0:2.0})


    def test_max_rel_error_dicts(self):
        self.assertEqual(max_rel_error_dicts({0:0.00139, 1:2.4, 2:87, 3:1, 4:97.8, 5:4.35}, 
                                             {0:1.00139, 1:1.4, 2:86, 3:2, 4:98.2, 5:33.35}), 
                         1/0.00139)

    def test_max_error_dicts(self):
        self.assertEqual(max_error_dicts({0:0.00139, 1:2.4, 2:87, 3:1, 4:97.8, 5:4.35}, 
                                             {0:1.00139, 1:1.4, 2:86, 3:2, 4:98.2, 5:33.35}), 
                         29)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)