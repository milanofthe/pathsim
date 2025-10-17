########################################################################################
##
##                                  TESTS FOR
##                               'blocks.noise.py'
##
##                              Milan Rother 2024/25
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.noise import WhiteNoise, PinkNoise


# TESTS ================================================================================

class TestWhiteNoise(unittest.TestCase):
    """
    Test the implementation of the 'WhiteNoise' block class
    """

    def test_init_default(self):
        """Test default initialization"""

        WN = WhiteNoise()

        self.assertEqual(WN.spectral_density, 1)
        self.assertEqual(WN.sampling_rate, None)
        self.assertEqual(WN.sigma, np.sqrt(1))
        self.assertEqual(WN.n_samples, 0)
        self.assertEqual(WN.noise, 0.0)
        self.assertEqual(len(WN.inputs), 0)  # Source block has no inputs


    def test_init_custom(self):
        """Test custom initialization"""

        WN = WhiteNoise(spectral_density=4.0, sampling_rate=10.0)

        self.assertEqual(WN.spectral_density, 4.0)
        self.assertEqual(WN.sampling_rate, 10.0)
        self.assertEqual(WN.sigma, np.sqrt(4.0))


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        WN = WhiteNoise()
        self.assertEqual(len(WN), 0)


    def test_sample_no_rate(self):
        """Test sampling without rate limit (every timestep)"""

        WN = WhiteNoise(spectral_density=1.0)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(5):
            WN.sample(t)
            WN.update(t)
            values.append(WN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)

        # Check that counter increased
        self.assertEqual(WN.n_samples, 5)


    def test_sample_with_rate(self):
        """Test sampling with specified rate"""

        WN = WhiteNoise(spectral_density=1.0, sampling_rate=1.0)

        # At t=0.6, should sample (0 < 0.6 * 1.0)
        WN.sample(0.6)
        first_value = WN.noise
        self.assertEqual(WN.n_samples, 1)

        # At t=0.7, should not sample (1 < 0.7 * 1.0 is False)
        WN.sample(0.7)
        self.assertEqual(WN.noise, first_value)
        self.assertEqual(WN.n_samples, 1)

        # At t=2.0, should sample again (1 < 2.0 * 1.0)
        WN.sample(2.0)
        self.assertEqual(WN.n_samples, 2)


    def test_update(self):
        """Test update method outputs current noise value"""

        WN = WhiteNoise()

        WN.noise = 5.5
        WN.update(0)

        self.assertEqual(WN.outputs[0], 5.5)


    def test_reset(self):
        """Test reset clears noise samples and counter"""

        WN = WhiteNoise()

        # Generate some samples
        for t in range(5):
            WN.sample(t)

        # Verify samples were generated
        self.assertEqual(WN.n_samples, 5)
        self.assertNotEqual(WN.noise, 0.0)

        # Reset
        WN.reset()

        # Check reset worked
        self.assertEqual(WN.n_samples, 0)
        self.assertEqual(WN.noise, 0.0)


class TestPinkNoise(unittest.TestCase):
    """
    Test the implementation of the 'PinkNoise' block class
    """

    def test_init_default(self):
        """Test default initialization"""

        PN = PinkNoise()

        self.assertEqual(PN.spectral_density, 1)
        self.assertEqual(PN.num_octaves, 16)
        self.assertEqual(PN.sampling_rate, None)
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(PN.noise, 0.0)
        self.assertEqual(len(PN.octave_values), 16)
        self.assertEqual(len(PN.inputs), 0)  # Source block has no inputs


    def test_init_custom(self):
        """Test custom initialization"""

        PN = PinkNoise(spectral_density=4.0, num_octaves=8, sampling_rate=10.0)

        self.assertEqual(PN.spectral_density, 4.0)
        self.assertEqual(PN.num_octaves, 8)
        self.assertEqual(PN.sampling_rate, 10.0)
        self.assertEqual(PN.sigma, np.sqrt(4.0/8))
        self.assertEqual(len(PN.octave_values), 8)


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        PN = PinkNoise()
        self.assertEqual(len(PN), 0)


    def test_sample_no_rate(self):
        """Test sampling without rate limit (every timestep)"""

        PN = PinkNoise(spectral_density=1.0, num_octaves=8)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(10):
            PN.sample(t)
            PN.update(t)
            values.append(PN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)

        # Check that counter increased
        self.assertEqual(PN.n_samples, 10)


    def test_sample_with_rate(self):
        """Test sampling with specified rate"""

        PN = PinkNoise(spectral_density=1.0, num_octaves=8, sampling_rate=1.0)

        # At t=0.6, should sample (0 < 0.6 * 1.0)
        PN.sample(0.6)
        self.assertEqual(PN.n_samples, 1)

        # At t=0.7, should not sample (1 < 0.7 * 1.0 is False)
        old_count = PN.n_samples
        PN.sample(0.7)
        self.assertEqual(PN.n_samples, old_count)

        # At t=2.0, should sample again (1 < 2.0 * 1.0)
        PN.sample(2.0)
        self.assertEqual(PN.n_samples, 2)


    def test_octave_update_algorithm(self):
        """Test that octaves are updated according to Voss-McCartney algorithm"""

        PN = PinkNoise(num_octaves=4)

        # Sample multiple times and check that octave values change
        initial_octaves = PN.octave_values.copy()

        for _ in range(10):
            PN.sample(0)  # Increment counter

        # At least some octave values should have changed
        self.assertFalse(np.array_equal(initial_octaves, PN.octave_values))


    def test_update(self):
        """Test update method outputs current noise value"""

        PN = PinkNoise()

        PN.noise = 3.7
        PN.update(0)

        self.assertEqual(PN.outputs[0], 3.7)


    def test_reset(self):
        """Test reset clears noise samples, counter, and resets octaves"""

        PN = PinkNoise(num_octaves=8)

        # Generate some samples
        initial_octaves = PN.octave_values.copy()
        for t in range(5):
            PN.sample(t)

        # Verify samples were generated
        self.assertEqual(PN.n_samples, 5)
        self.assertNotEqual(PN.noise, 0.0)

        # Reset
        PN.reset()

        # Check reset worked
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(PN.noise, 0.0)

        # Octave values should be reinitialized (different from before)
        # (technically could be same by chance, but very unlikely)
        self.assertEqual(len(PN.octave_values), 8)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
