########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.adder.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.adder import Adder

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestAdder(unittest.TestCase):
    """
    Test the implementation of the 'Adder' block class
    """

    def test_init(self):
        
        #default initialization
        A = Adder()
        self.assertEqual(A.operations, None)

        #input validation
        for a in [0.4, 3, [1, -1, 0], "a", "10"]:
            with self.assertRaises(ValueError):
                A = Adder(a)

        #special initialization
        A = Adder("+-")
        self.assertEqual(A.operations, "+-")


    def test_embedding(self):
        """test algebraic components via embedding"""

        A = Adder()

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) + np.sin(t) + 3.0 + t

        E = Embedding(A, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        A = Adder("+-")

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) - np.sin(t) 

        E = Embedding(A, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        A = Adder("++-0")

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) + np.sin(t) - 3

        E = Embedding(A, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_linearization(self):
        """test linearization and delinearization"""

        A = Adder()

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) + np.sin(t) + 3.0 + t

        E = Embedding(A, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        #linearize block
        A.linearize(3)

        for t in range(10): 
            a, b = E.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #linearize at differnt point in time block
        A.linearize(12)

        for t in range(10): 
            a, b = E.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        A.delinearize()

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


        A = Adder("+-0+")

        def src(t): return np.cos(t), np.sin(t), 3.0, t
        def ref(t): return np.cos(t) - np.sin(t) + t

        E = Embedding(A, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        #linearize block
        A.linearize(3)

        for t in range(10): 
            a, b = E.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #linearize at differnt point in time block
        A.linearize(12)

        for t in range(10): 
            a, b = E.check_MIMO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        A.delinearize()

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_sensitivity(self):
        """test compatibility with AD framework"""

        from pathsim.optim.value import Value


        a, b, c = Value.array([3.2, -0.3, 1000])

        A = Adder()

        def src(t): return a, b, c
        def ref(t): return a + b + c

        E = Embedding(A, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        for t in range(10): 
            y, _ = E.check_MIMO(t)
            self.assertEqual(Value.der(y, a), 1.0)
            self.assertEqual(Value.der(y, b), 1.0)
            self.assertEqual(Value.der(y, c), 1.0)


        A = Adder("+-0")

        def src(t): return a, b, c
        def ref(t): return a - b

        E = Embedding(A, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))

        for t in range(10): 
            y, _ = E.check_MIMO(t)
            self.assertEqual(Value.der(y, a), 1.0)
            self.assertEqual(Value.der(y, b), -1.0)
            self.assertEqual(Value.der(y, c), 0.0)
        

    def test_update_single(self):
        
        A = Adder()

        #set block inputs
        A.inputs[0] = 1

        #update block
        A.update(None)

        #test if update was correct
        self.assertEqual(A.outputs[0], 1)


    def test_update_multi(self):

        A = Adder()

        #set block inputs
        A.inputs[0] = 1
        A.inputs[1] = 2.0
        A.inputs[2] = 3.1

        #update block
        A.update(None)

        #test if update was correct
        self.assertEqual(A.outputs[0], 6.1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)