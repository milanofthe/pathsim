import unittest
import skrf as rf
import matplotlib.pyplot as plt

from numpy.testing import assert_allclose

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

    def test_ABCD(self):
        "Test space-state ABCD parameters."
        # ABCD(E) arrays from scikit-rf
        ntwk = rf.data.ring_slot_meas
        vf = rf.VectorFitting(ntwk)
        vf.auto_fit()
        A, B, C, D, E = vf._get_ABCDE()

        # ABCD arrays from pathsim
        rfblock = RFNetwork(ntwk)

        assert_allclose(rfblock.A, A)
        assert_allclose(rfblock.B, B)
        assert_allclose(rfblock.C, C)
        assert_allclose(rfblock.D, D)

    def test_s_parameters(self):
        "Test S-parameters deduced from ABCD parameters."
        # original network
        ntwk = rf.data.ring_slot_meas
        # Get S-parameter from pathsim ABCD parameters
        rfblock = RFNetwork(ntwk)
        s = rf.VectorFitting._get_s_from_ABCDE(freqs=ntwk.f, A=rfblock.A, B=rfblock.B, C=rfblock.C, D=rfblock.D, E=0)
        # check equality (with a large tolerance, since it's measurements vs VF model)
        assert_allclose(ntwk.s, s, atol=0.05)

class TestTwoPort(unittest.TestCase):
    def test_init(self):
        two_port = RFNetwork(rf.data.ring_slot)
        print(two_port)