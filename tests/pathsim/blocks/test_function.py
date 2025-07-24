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

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestFunction(unittest.TestCase):
    """
    Test the implementation of the 'Function' block class
    """

    def test_init(self):

        def f(a): return a**2
        
        F = Function(func=f)

        #test if the function works
        self.assertEqual(F.func(1), f(1))
        self.assertEqual(F.func(2), f(2))
        self.assertEqual(F.func(3), f(3))

        #test input validation
        for v in [2, 0.3, 1j, np.ones(3)]:
            with self.assertRaises(ValueError): 
                F = Function(func=v)


    def test_embedding_siso(self):

        def f(a): return a**2
        
        F = Function(func=f)
        
        def src(t): return np.sin(t)
        def ref(t): return np.sin(t)**2
        
        E = Embedding(F, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        def src(t): return np.tanh(t)
        def ref(t): return np.tanh(t)**2

        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_embedding_miso(self):

        def f(a, b, c): return a**2 + b - c
        
        F = Function(func=f)

        def src(t): return np.sin(t), np.cos(t), np.tanh(t)
        def ref(t): return np.sin(t)**2 + np.cos(t) - np.tanh(t)
        
        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))



    def test_linearization_miso(self):
        """test linearization and delinearization"""

        def f(a, b, c): return a**2 + b - c
        
        F = Function(func=f)

        def src(t): return np.cos(t), t, 3.0
        def ref(t): return np.cos(t)**2 + t - 3.0 

        E = Embedding(F, src, ref)

        for t in range(10): 
            self.assertEqual(*E.check_MIMO(t))

        #linearize block
        F.linearize(t)

        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        F.delinearize()

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_linearization_mimo(self):
        """test linearization and delinearization"""

        def f(a, b, c): return a**2 + b - c, 4*np.sin(b*c)
        
        F = Function(func=f)

        def src(t): return np.cos(t), t, 3.0
        def ref(t): return np.cos(t)**2 + t - 3.0, 4*np.sin(t*3) 

        E = Embedding(F, src, ref)

        for t in range(10): 
            y, r = E.check_MIMO(t)
            self.assertEqual(y[0], r[0])
            self.assertEqual(y[1], r[1])

        #linearize block
        F.linearize(t)

        y, r = E.check_MIMO(t)
        self.assertEqual(y[0], r[0])
        self.assertEqual(y[1], r[1])

        #delinearize
        F.delinearize()

        for t in range(10): 
            y, r = E.check_MIMO(t)
            self.assertEqual(y[0], r[0])
            self.assertEqual(y[1], r[1])


    def test_sensitivity(self):
        """test compatibility with AD framework"""

        from pathsim.optim.value import Value

        a, b, c = Value.array([3, 0.5, -1])

        def f(a, b, c): return a**2 + b - c
        
        F = Function(func=f)

        def src(t): return a, b, c
        def ref(t): return 10.5

        E = Embedding(F, src, ref)

        y, r = E.check_MIMO(0)
        self.assertEqual(y, r)

        #check derivatives
        self.assertEqual(Value.der(y, a), 6)
        self.assertEqual(Value.der(y, b), 1)
        self.assertEqual(Value.der(y, c), -1)




    def test_update_siso(self):

        def f(a):
            return a**2
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], f(3))


    def test_update_miso(self):

        def f(a, b, c):
            return a**2 + b - c
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3
        F.inputs[1] = 2
        F.inputs[2] = 1

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], f(3, 2, 1))


    def test_update_simo(self):

        def f(a):
            return a**2, 2*a, 1
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], 9)
        self.assertEqual(F.outputs[1], 6)
        self.assertEqual(F.outputs[2], 1)


    def test_update_mimo(self):

        def f(a, b, c):
            return a**2-b, 3*c
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3
        F.inputs[1] = 2
        F.inputs[2] = 1

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], 7)
        self.assertEqual(F.outputs[1], 3)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)