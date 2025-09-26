########################################################################################
##
##                                  TESTS FOR 
##                       'blocks.tritium.residencetime.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.tritium import ResidenceTime, Process

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestFusionResidenceTime(unittest.TestCase):
    """
    Test the implementation of the 'ResidenceTime' block class from the fusion toolbox

    The block inherits from `ODE`
    """
    
    def test_init(self):

        #default initialization
        R = ResidenceTime()

        self.assertEqual(R.tau, 1)
        self.assertEqual(R.betas, 1)
        self.assertEqual(R.gammas, 1)
        self.assertEqual(R.source_term, 0)

        #input validation
        with self.assertRaises(ValueError):
            R = ResidenceTime(tau=0)

        #specific initialization
        R = ResidenceTime(tau=0.1, betas=[1, 3, 5], gammas=[2, 4], initial_value=10, source_term=11)

        self.assertEqual(R.tau,0.1)
        self.assertTrue(np.allclose(R.betas, np.array([1, 3, 5])))
        self.assertTrue(np.allclose(R.gammas, np.array([2, 4])))

        #set solver to check internal solver instance
        R.set_solver(EUF, parent=None)

        self.assertTrue(R.engine)
        self.assertEqual(R.engine.initial_value, 10)


    def test_update(self):

        #default
        R = ResidenceTime()
        R.set_solver(EUF, parent=None)

        R.update(None)

        self.assertEqual(R.outputs[0], 0)

        #specific
        R = ResidenceTime(tau=0.1, betas=[1, 3, 5], gammas=[2, 4], initial_value=11, source_term=12)
        R.set_solver(EUF, parent=None)

        R.update(None)

        self.assertEqual(R.outputs[0], 22) # initial_value * gammas[0]
        self.assertEqual(R.outputs[1], 44) # initial_value * gammas[1]



class TestFusionProcess(unittest.TestCase):
    """
    Test the implementation of the 'Process' block class from the fusion toolbox.

    This block inherits from `ResidenceTime`, just testing the initialization.
    """

    def test_init(self):

        #default initialization
        P = Process()

        self.assertEqual(P.tau, 1)
        self.assertEqual(P.betas, 1)
        self.assertTrue(np.allclose(P.gammas, [1, 1])) # 1, 1/tau
        self.assertEqual(P.source_term, 0)

        #specific initialization
        P = Process(tau=0.1, source_term=33, initial_value=44)

        self.assertEqual(P.tau,0.1)
        self.assertEqual(P.betas, 1)
        self.assertTrue(np.allclose(P.gammas, np.array([1, 10])))
        self.assertEqual(P.source_term, 33)

        #set solver to check internal solver instance
        P.set_solver(EUF, parent=None)

        self.assertTrue(P.engine)
        self.assertEqual(P.engine.initial_value, 44)






# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
