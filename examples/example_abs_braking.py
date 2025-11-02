#########################################################################################
##
##                      PathSim Anti-lock Braking System (ABS) Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Function, Scope, Clip, Constant
from pathsim.solvers import RKCK54
from pathsim.events import ZeroCrossing


# VEHICLE AND TIRE PARAMETERS ===========================================================

# Vehicle parameters
M = 1500         # Vehicle mass [kg]
R = 0.3          # Wheel radius [m]
J_w = 1.0        # Wheel rotational inertia [kg·m²]
F_z = M * 9.81 / 4  # Normal force per wheel [N]

# Tire friction model (Pacejka "Magic Formula")
B_coef = 10.0    # Stiffness factor
C_coef = 1.9     # Shape factor
D_coef = 1.0     # Peak friction coefficient

# ABS control parameters
lambda_opt = 0.15      # Optimal slip ratio
abs_threshold = 0.02   # Control band around optimal

# Brake torque
T_brake = 2000   # Maximum brake torque [N·m]

# Initial conditions
v0 = 30          # Initial vehicle speed [m/s]
omega0 = v0 / R  # Initial wheel angular velocity [rad/s]


# FRICTION AND SLIP MODELS ==============================================================

def friction_coefficient(slip):
    """Pacejka tire friction model"""
    return D_coef * np.sin(C_coef * np.arctan(B_coef * slip))

def calculate_slip(v, omega):
    """Calculate slip ratio: lambda = (v - R*omega) / v"""
    omega_actual = max(0, omega)
    if v < 0.1:
        return 0.0
    slip = (v - R * omega_actual) / v
    return np.clip(slip, 0, 1)

def friction_force(slip_ratio):
    """Tire friction force"""
    return friction_coefficient(slip_ratio) * F_z

# ABS control state
abs_state = {'apply_brake': True}

def abs_control():
    """ABS bang-bang controller using state"""
    return T_brake if abs_state['apply_brake'] else 0


# SYSTEM SETUP (WITH ABS) ===============================================================

# Wheel dynamics: J_w * domega/dt = -T_brake + R * F_x
omg_raw = Integrator(omega0)
omg_clp = Clip(min_val=0, max_val=1000)
omg_acc = Amplifier(1/J_w)

# Vehicle dynamics: M * dv/dt = -F_x
vel = Integrator(v0)
vel_acc = Amplifier(-1/M)

# Slip and friction
slp = Function(calculate_slip)
frc = Function(friction_force)
frc_cof = Function(friction_coefficient)

# ABS control
brk = Constant(T_brake)  # Will be modulated by events
brk_neg = Amplifier(-1)

# Torque summation
whl_trq = Amplifier(R)
trq_sum = Adder("++")

# Measurement
whl_vel = Amplifier(R)
sco1 = Scope(labels=["Vehicle Speed [m/s]", "Wheel Speed [m/s]"])
sco2 = Scope(labels=["Slip Ratio", "Friction Coeff"])

blocks = [
    omg_raw, omg_clp, omg_acc, vel, vel_acc,
    slp, frc, frc_cof, brk, brk_neg,
    whl_trq, trq_sum, whl_vel, sco1, sco2
]


# CONNECTIONS ===========================================================================

connections = [
    # Wheel dynamics
    Connection(omg_acc, omg_raw),
    Connection(omg_raw, omg_clp),
    Connection(omg_clp, whl_vel),
    Connection(trq_sum, omg_acc),

    # Vehicle dynamics
    Connection(vel_acc, vel),
    Connection(vel, sco1[0]),

    # Slip calculation
    Connection(vel, slp[0]),
    Connection(omg_clp, slp[1]),
    Connection(slp, sco2[0]),

    # Friction
    Connection(slp, frc),
    Connection(frc, whl_trq, vel_acc),
    Connection(slp, frc_cof),
    Connection(frc_cof, sco2[1]),

    # Brake control (ABS with events)
    Connection(brk, brk_neg),
    Connection(brk_neg, trq_sum[1]),

    # Torque balance
    Connection(whl_trq, trq_sum[0]),
    Connection(whl_vel, sco1[1]),
]


# ABS CONTROL EVENTS ====================================================================

# Event: slip too high -> release brake
def evt_slip_high(t):
    """Detects when slip exceeds upper threshold"""
    slip_val = slp.outputs[0]
    return slip_val - (lambda_opt + abs_threshold)

def act_release_brake(t):
    """Release brake when slip is too high"""
    abs_state['apply_brake'] = False
    brk.value = 0

evt_high = ZeroCrossing(func_evt=evt_slip_high, func_act=act_release_brake, tolerance=1e-4)

# Event: slip too low -> apply brake
def evt_slip_low(t):
    """Detects when slip falls below lower threshold"""
    slip_val = slp.outputs[0]
    return (lambda_opt - abs_threshold) - slip_val

def act_apply_brake(t):
    """Apply brake when slip is too low"""
    abs_state['apply_brake'] = True
    brk.value = T_brake

evt_low = ZeroCrossing(func_evt=evt_slip_low, func_act=act_apply_brake, tolerance=1e-4)

events = [evt_high, evt_low]


# SIMULATION ============================================================================

Sim = Simulation(blocks, connections, events, Solver=RKCK54, dt=0.001)


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run simulation
    Sim.run(5)

    # Plot results
    sco1.plot(lw=2)
    sco2.plot(lw=2)

    plt.show()
