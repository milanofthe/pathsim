#########################################################################################
##
##                    PathSim Example: Kalman Filter State Estimation
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Constant,
    Integrator,
    Adder,
    WhiteNoise,
    KalmanFilter,
    Scope
)


# KALMAN FILTER FOR POSITION/VELOCITY TRACKING =========================================

# Simulation parameters
dt = 0.05  # timestep

# True system: object moving with constant velocity
v_true = 2.0  # m/s
x0_true = 0.0  # initial position

# Measurement noise characteristics
measurement_std = 0.2  # standard deviation of position sensor noise

# Kalman filter parameters
F = np.array([[1, dt], [0, 1]])        # state transition (constant velocity model)
H = np.array([[1, 0]])                 # measurement matrix (measure position only)
Q = np.diag([0.01, 0.01])              # process noise covariance
R = np.array([[measurement_std**2]])   # measurement noise covariance
x0_kf = np.array([0, 0])               # initial estimate [position, velocity]
P0_kf = np.eye(2) * 5                  # initial covariance

# Build the system -----------------------------------------------------------------------

# True system
vel = Constant(v_true)
pos = Integrator(x0_true)

# Noisy measurement
noise = WhiteNoise(spectral_density=measurement_std**2)
measured_pos = Adder()

# Kalman filter
kf = KalmanFilter(F, H, Q, R, x0=x0_kf, P0=P0_kf)

# Scopes for recording
sc_true = Scope(labels=["true position", "true velocity"])
sc_meas = Scope(labels=["measured position"])
sc_est = Scope(labels=["estimated position", "estimated velocity"])

blocks = [vel, pos, noise, measured_pos, kf, sc_true, sc_meas, sc_est]

# Connections
connections = [
    Connection(vel, pos, sc_true[1]),
    Connection(pos, measured_pos[0], sc_true[0]),
    Connection(noise, measured_pos[1]),
    Connection(measured_pos, kf, sc_meas),
    Connection(kf[0], sc_est[0]),
    Connection(kf[1], sc_est[1])
]

# Initialize simulation
Sim = Simulation(
    blocks,
    connections,
    dt=dt,
)


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run the simulation
    Sim.run(duration=20)

    # Read data from scopes
    t_true, [pos_true, vel_true] = sc_true.read()
    t_meas, [pos_meas] = sc_meas.read()
    t_est, [pos_est, vel_est] = sc_est.read()

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True, dpi=200)

    # Position comparison
    ax1.set_title('Kalman Filter: Position and Velocity Estimation')
    ax1.plot(t_meas, pos_meas, ".", label='Noisy Measurement')
    ax1.plot(t_true, pos_true, "-", label='True Position')
    ax1.plot(t_est, pos_est, "--", label='Kalman Estimate')
    ax1.set_ylabel('Position [m]')
    ax1.legend()

    # Velocity comparison
    ax2.plot(t_true, vel_true, "-", label='True Velocity')
    ax2.plot(t_est, vel_est, "--", label='Kalman Estimate')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_xlabel('Time [s]')
    ax2.legend()

    # Calculate estimation errors
    pos_error = np.abs(pos_est - pos_true)
    vel_error = np.abs(vel_est - vel_true)

    # Estimation error over time
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True, dpi=200)

    ax3.plot(t_est, pos_error)
    ax3.set_ylabel('Position Error [m]')
    ax3.set_title('Kalman Filter Estimation Error')

    ax4.plot(t_est, vel_error)
    ax4.set_ylabel('Velocity Error [m/s]')
    ax4.set_xlabel('Time [s]')

    plt.show()