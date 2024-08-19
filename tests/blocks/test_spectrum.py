########################################################################################
##
##                                  TESTS FOR 
##                             'blocks.spectrum.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.spectrum import Spectrum

#base solver for testing
from pathsim.solvers._solver import Solver 


# TESTS ================================================================================

class TestSpectrum(unittest.TestCase):
    """
    Test the implementation of the 'Spectrum' block class
    """

    def test_init(self):

        #test default initialization
        S = Spectrum()
        self.assertEqual(S.time, 0.0)
        self.assertEqual(S.t_wait, 0.0)
        self.assertEqual(S.alpha, 0.0)
        self.assertEqual(S.labels, [])
        self.assertEqual(len(S.freq), 0)
        self.assertEqual(len(S.omega), 0)

        #test specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq, t_wait=20, alpha=0.01, labels=["1", "2"])
        self.assertEqual(S.time, 0.0)
        self.assertEqual(S.t_wait, 20)
        self.assertEqual(S.alpha, 0.01)
        self.assertEqual(S.labels, ["1", "2"])
        self.assertTrue(np.all(S.freq == _freq))
        self.assertTrue(np.all(S.omega == 2*np.pi*_freq))


    def test_len(self):

        S = Spectrum()

        #no direct passthrough
        self.assertEqual(len(S), 0)


    def test_str(self):

        S = Spectrum()

        #test default str method
        self.assertEqual(str(S), "Spectrum")


    def test_set_solver(self):

        S = Spectrum()

        #test that no solver is initialized
        self.assertEqual(S.engine, None)

        S.set_solver(Solver, tolerance_lte=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(S.engine, Solver))
        self.assertEqual(S.engine.tolerance_lte, 1e-6)
        self.assertEqual(S.engine.initial_value, 0.0)

        S.set_solver(Solver, tolerance_lte=1e-3)

        #test that solver tolerance is changed
        self.assertEqual(S.engine.tolerance_lte, 1e-3)


    def test_read(self):

        #test read for no engine and default initialization
        S = Spectrum()

        freq, spec = S.read()
        
        self.assertEqual(len(freq), 0)
        self.assertEqual(len(spec), 0)

        #test read for no engine and specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq)

        freq, spec = S.read()
        
        self.assertTrue(np.all(freq == _freq))
        self.assertTrue(np.all(spec == np.zeros(100)))

        #test read for engine and specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq)
        S.set_solver(Solver)

        freq, spec = S.read()
        
        self.assertTrue(np.all(freq == _freq))
        self.assertTrue(np.all(spec == np.zeros(100)))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)