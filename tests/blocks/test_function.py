########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.function.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.function import Function


# TESTS ================================================================================

class TestFunction(unittest.TestCase):
    """
    Test the implementation of the 'Function' block class
    """

    def test_init(self):

        def f(a):
            return a**2
        
        F = Function(func=f)

        #test if the function works
        self.assertEqual(F.func(1), f(1))
        self.assertEqual(F.func(2), f(2))
        self.assertEqual(F.func(3), f(3))

        #test input validation
        with self.assertRaises(ValueError): 
            F = Function(func=2)


    def test_str(self):

        F = Function()

        #test default str method
        self.assertEqual(str(F), "Function")


    def test_update_siso(self):

        def f(a):
            return a**2
        
        F = Function(func=f)

        #set block inputs
        F.set(0, 3)

        #update block
        err = F.update(None)

        #test if update was correct
        self.assertEqual(F.get(0), f(3))

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = F.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


    def test_update_miso(self):

        def f(a, b, c):
            return a**2 + b - c
        
        F = Function(func=f)

        #set block inputs
        F.set(0, 3)
        F.set(1, 2)
        F.set(2, 1)

        #update block
        err = F.update(None)

        #test if update was correct
        self.assertEqual(F.get(0), f(3, 2, 1))

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = F.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


    def test_update_simo(self):

        def f(a):
            return a**2, 2*a, 1
        
        F = Function(func=f)

        #set block inputs
        F.set(0, 3)

        #update block
        err = F.update(None)

        #test if update was correct
        self.assertEqual(F.get(0), 9)
        self.assertEqual(F.get(1), 6)
        self.assertEqual(F.get(2), 1)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = F.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


    def test_update_mimo(self):

        def f(a, b, c):
            return a**2-b, 3*c
        
        F = Function(func=f)

        #set block inputs
        F.set(0, 3)
        F.set(1, 2)
        F.set(2, 1)

        #update block
        err = F.update(None)

        #test if update was correct
        self.assertEqual(F.get(0), 7)
        self.assertEqual(F.get(1), 3)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = F.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)