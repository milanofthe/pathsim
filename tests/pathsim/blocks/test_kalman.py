########################################################################################
##
##                                  TESTS FOR
##                              'blocks.kalman.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.kalman import KalmanFilter


# TESTS ================================================================================

class TestKalmanFilter(unittest.TestCase):
    """
    Test the implementation of the 'KalmanFilter' block class
    """

    def test_init_basic(self):
        """Test basic initialization without optional parameters."""

        # Simple 2D system (position and velocity)
        F = np.array([[1, 0.1], [0, 1]])  # state transition
        H = np.array([[1, 0]])             # measure position only
        Q = np.eye(2) * 0.01               # process noise
        R = np.array([[0.1]])              # measurement noise

        kf = KalmanFilter(F, H, Q, R)

        # Check dimensions
        self.assertEqual(kf.n, 2)  # state dimension
        self.assertEqual(kf.m, 1)  # measurement dimension
        self.assertEqual(kf.p, 0)  # no control input

        # Check default initial conditions
        np.testing.assert_array_equal(kf.x, np.zeros(2))
        np.testing.assert_array_equal(kf.P, np.eye(2))

        # Check matrices
        np.testing.assert_array_equal(kf.F, F)
        np.testing.assert_array_equal(kf.H, H)
        np.testing.assert_array_equal(kf.Q, Q)
        np.testing.assert_array_equal(kf.R, R)
        self.assertIsNone(kf.B)

        # Check io dimensions
        self.assertEqual(len(kf.inputs), 1)  # m measurements
        self.assertEqual(len(kf.outputs), 2)  # n states


    def test_init_with_initial_conditions(self):
        """Test initialization with custom initial state and covariance."""

        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2) * 5.0

        kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)

        np.testing.assert_array_equal(kf.x, x0)
        np.testing.assert_array_equal(kf.P, P0)


    def test_init_with_control_input(self):
        """Test initialization with control input matrix B."""

        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        B = np.array([[0], [0.1]])  # control affects velocity

        kf = KalmanFilter(F, H, Q, R, B=B)

        # Check dimensions
        self.assertEqual(kf.n, 2)
        self.assertEqual(kf.m, 1)
        self.assertEqual(kf.p, 1)  # one control input

        np.testing.assert_array_equal(kf.B, B)

        # Check io dimensions (m measurements + p controls)
        self.assertEqual(len(kf.inputs), 2)  # 1 measurement + 1 control
        self.assertEqual(len(kf.outputs), 2)  # n states


    def test_init_with_dt(self):
        """Test initialization with discrete time step."""

        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        dt = 0.1

        kf = KalmanFilter(F, H, Q, R, dt=dt)

        self.assertEqual(kf.dt, dt)
        self.assertEqual(len(kf.events), 1)  # scheduled event created


    def test_len(self):
        """Test that KalmanFilter has no passthrough."""

        F = np.eye(2)
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        kf = KalmanFilter(F, H, Q, R)

        # No direct passthrough
        self.assertEqual(len(kf), 0)


    def test_constant_position_estimation(self):
        """Test Kalman filter on constant position system."""

        # System: stationary object at position x=5.0
        F = np.array([[1.0]])  # position doesn't change
        H = np.array([[1.0]])  # measure position directly
        Q = np.array([[0.0]])  # no process noise
        R = np.array([[1.0]])  # measurement noise variance = 1.0

        # Start with uncertain initial estimate
        x0 = np.array([0.0])
        P0 = np.array([[10.0]])

        kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)

        # Simulate measurements of true position = 5.0
        true_position = 5.0
        measurements = [true_position] * 10

        for z in measurements:
            kf.inputs[0] = z
            kf._kf_update()

        # After 10 measurements, estimate should be close to true value
        self.assertAlmostEqual(kf.x[0], true_position, delta=0.5)

        # Covariance should decrease (more certain)
        self.assertLess(kf.P[0, 0], P0[0, 0])


    def test_constant_velocity_estimation(self):
        """Test Kalman filter on constant velocity system."""

        dt = 0.1

        # System: constant velocity model
        F = np.array([[1, dt], [0, 1]])    # [position, velocity]
        H = np.array([[1, 0]])              # measure position only
        Q = np.diag([0.001, 0.001])         # small process noise
        R = np.array([[0.5]])               # measurement noise

        # True system state
        true_position = 0.0
        true_velocity = 2.0

        # Initial estimate (uncertain)
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 2.0

        kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)

        # Simulate system over time
        num_steps = 50
        for k in range(num_steps):
            # True position at this time
            true_position += true_velocity * dt

            # Noisy measurement
            measurement = true_position

            # Update filter
            kf.inputs[0] = measurement
            kf._kf_update()

        # After convergence, velocity estimate should be close to true velocity
        self.assertAlmostEqual(kf.x[1], true_velocity, delta=0.5)

        # Position should also track
        self.assertAlmostEqual(kf.x[0], true_position, delta=1.0)


    def test_with_control_input(self):
        """Test Kalman filter with control input."""

        dt = 0.1

        # System with control input (force applied to mass)
        F = np.array([[1, dt], [0, 1]])     # [position, velocity]
        H = np.array([[1, 0]])               # measure position
        B = np.array([[0.5*dt*dt], [dt]])    # control matrix (acceleration)
        Q = np.diag([0.01, 0.01])
        R = np.array([[0.1]])

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        kf = KalmanFilter(F, H, Q, R, B=B, x0=x0, P0=P0)

        # Apply constant control input (acceleration = 1.0)
        control = 1.0
        num_steps = 20

        position = 0.0
        velocity = 0.0

        for k in range(num_steps):
            # True system evolution with control
            velocity += control * dt
            position += velocity * dt

            # Measurement
            measurement = position

            # Update filter with measurement and control
            kf.inputs[0] = measurement  # measurement
            kf.inputs[1] = control       # control input
            kf._kf_update()

        # State estimates should track the true values
        self.assertAlmostEqual(kf.x[0], position, delta=1.0)
        self.assertAlmostEqual(kf.x[1], velocity, delta=0.5)


    def test_sample_method_without_dt(self):
        """Test that sample method triggers update when dt is None."""

        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        x0 = np.array([0.0])

        kf = KalmanFilter(F, H, Q, R, x0=x0, dt=None)

        # Set measurement
        kf.inputs[0] = 5.0

        # Initial state
        initial_state = kf.x.copy()

        # Call sample
        kf.sample(0.0, 1)

        # State should have been updated
        self.assertNotEqual(kf.x[0], initial_state[0])

        # Output should be updated
        self.assertEqual(kf.outputs[0], kf.x[0])


    def test_output_update(self):
        """Test that outputs are correctly updated after filter update."""

        F = np.array([[1, 0.1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        x0 = np.array([1.0, 2.0])

        kf = KalmanFilter(F, H, Q, R, x0=x0)

        # Set measurement
        kf.inputs[0] = 3.0

        # Perform update
        kf._kf_update()

        # Outputs should match state
        np.testing.assert_array_equal(kf.outputs.to_array(), kf.x)


    def test_multidimensional_measurement(self):
        """Test Kalman filter with multiple measurements."""

        # 3D state: [x, y, vx, vy] - 2D position and velocity
        dt = 0.1
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measure both x and y position
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        Q = np.eye(4) * 0.01
        R = np.eye(2) * 0.1
        x0 = np.zeros(4)

        kf = KalmanFilter(F, H, Q, R, x0=x0)

        # Check dimensions
        self.assertEqual(kf.n, 4)  # state dimension
        self.assertEqual(kf.m, 2)  # measurement dimension
        self.assertEqual(len(kf.inputs), 2)   # 2 measurements
        self.assertEqual(len(kf.outputs), 4)  # 4 states

        # Perform a few updates
        for i in range(10):
            kf.inputs[0] = i * 0.1  # x measurement
            kf.inputs[1] = i * 0.2  # y measurement
            kf._kf_update()

        # Just check that it runs without error and produces reasonable output
        self.assertTrue(np.all(np.isfinite(kf.x)))
        self.assertTrue(np.all(np.isfinite(kf.P)))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
