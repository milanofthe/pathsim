# tests require scikit-rf package
import pytest, sys, os

try:
    import skrf as rf
except ImportError as e:
    pass

if "skrf" not in sys.modules:
    pytest.skip(allow_module_level=True)

import unittest
import numpy as np

from numpy.testing import assert_allclose
from pathlib import Path
from pathsim.blocks.rf import RFNetwork

def is_equal(ss1, ss2):
    """
    Test if two StateSpace blocks are equal.

    Parameters
    ----------
    ss1: StateSpace object
    ss2: StateSpace object

    Returns
    -------
    bool

    """
    if type(ss1) != type(ss2):
        return False
    for prop in ['A', 'B', 'C', 'D']:
        if not np.all(np.abs(getattr(ss1, prop) - getattr(ss2, prop)) < 1e-6):
            return False
    return True

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
        test_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        # skrf Network
        one_port1 = RFNetwork(rf.data.ring_slot_meas)
        # string
        one_port2 = RFNetwork(test_dir + 'ring_slot_meas.s1p')
        # Path
        one_port3 = RFNetwork(Path(test_dir + 'ring_slot_meas.s1p'))
        assert is_equal(one_port1, one_port2)
        assert is_equal(one_port1, one_port3)

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
        s = rfblock.s(ntwk.f)
        # check equality (with a large tolerance, since it's measurements vs VF model)
        assert_allclose(ntwk.s, s, atol=0.05)


class TestTwoPort(unittest.TestCase):
    def test_init(self):
        test_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        # skrf Network
        two_port1 = RFNetwork(rf.data.ring_slot)
        # string
        two_port2 = RFNetwork(test_dir + 'ring_slot.s2p')
        # Path
        two_port3 = RFNetwork(Path(test_dir + 'ring_slot.s2p'))
        assert is_equal(two_port1, two_port2)
        assert is_equal(two_port1, two_port3)

    def test_s_parameters(self):
        "Test S-parameters deduced from ABCD parameters for two-port Network."
        # original network
        ntwk = rf.data.ring_slot
        two_port = RFNetwork(ntwk)
        # State-space model
        rfblock = RFNetwork(ntwk)
        # S-parameter from ABCD parameters
        s = rfblock.s(ntwk.f)
        # check equality between original s parameters and reconstructed from vector fitting/state-space
        assert_allclose(ntwk.s, s, atol=1e-5)
