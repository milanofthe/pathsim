import unittest
import skrf as rf
from contourpy.chunk import two_factors

from pathsim.blocks.rf import RFNetwork

class TestSkrf(unittest.TestCase):
    """
    Test that scikit-rf is properly installed and usable.
    """
    def test_vf(self):
        """
        Test that skrf vector fitting runs correctly.
        """
        # skrf VF test case reproduced here
        nw = rf.data.ring_slot
        vf = rf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

class TestOnePort(unittest.TestCase):
    def test_init(self):
        one_port = RFNetwork(rf.data.ring_slot_meas)
        print(one_port)

class TestTwoPort(unittest.TestCase):
    def test_init(self):
        two_port = RFNetwork(rf.data.ring_slot)
        print(two_port)