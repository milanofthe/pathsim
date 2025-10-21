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
dt = 0.01  # timestep

# True system: object moving with constant velocity
v_true = 2.0  # m/s
x0_true = 0.0  # initial position

# Measurement noise characteristics
measurement_std = 0.6  # standard deviation of position sensor noise

# Kalman filter parameters
F = np.array([[1, dt], [0, 1]])        # state transition (constant velocity model)
H = np.array([[1, 0]])                 # measurement matrix (measure position only)
Q = np.diag([0.01, 0.01])              # process noise covariance
R = np.array([[measurement_std**2]])   # measurement noise covariance
x0_kf = np.array([0, 0])               # initial estimate [position, velocity]
P0_kf = np.eye(2) * 1                  # initial covariance

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True, dpi=120)

    # Position comparison
    ax1.plot(t_true, pos_true, 'g-', lw=2.5, label='True Position', zorder=3)
    ax1.plot(t_meas, pos_meas, 'r.', alpha=0.4, markersize=4, label='Noisy Measurement')
    ax1.plot(t_est, pos_est, 'b-', lw=2, label='Kalman Estimate', zorder=2)
    ax1.set_ylabel('Position [m]', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Kalman Filter: Position and Velocity Estimation', fontsize=12, fontweight='bold')

    # Velocity comparison
    ax2.plot(t_true, vel_true, 'g-', lw=2.5, label='True Velocity', zorder=3)
    ax2.plot(t_est, vel_est, 'b-', lw=2, label='Kalman Estimate', zorder=2)
    ax2.axhline(v_true, color='gray', linestyle='--', alpha=0.5, label='Target Velocity')
    ax2.set_ylabel('Velocity [m/s]', fontsize=11)
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Calculate and display estimation errors
    pos_error = np.abs(pos_est - pos_true)
    vel_error = np.abs(vel_est - vel_true)

    # Estimation error over time
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True, dpi=120)
    
    ax3.plot(t_est, pos_error, 'b-', lw=1.5)
    ax3.fill_between(t_est, 0, pos_error, alpha=0.3)
    ax3.set_ylabel('Position Error [m]', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Kalman Filter Estimation Error', fontsize=12, fontweight='bold')
    
    ax4.plot(t_est, vel_error, 'b-', lw=1.5)
    ax4.fill_between(t_est, 0, vel_error, alpha=0.3)
    ax4.set_ylabel('Velocity Error [m/s]', fontsize=11)
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.show()