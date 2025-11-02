#########################################################################################
##
##                           PathSim DC Motor Speed Control Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import StepSource, Integrator, Amplifier, Adder, Scope, AntiWindupPID, Clip
from pathsim.solvers import RKCK54


# DC MOTOR PARAMETERS ===================================================================

# Electrical parameters
R = 1.0          # Armature resistance [Ohm]
L = 0.001        # Armature inductance [H]
K_e = 0.1        # Back-EMF constant [V·s/rad]

# Mechanical parameters
J = 0.01         # Rotor inertia [kg·m²]
B = 0.001        # Viscous friction [N·m·s/rad]
K_t = 0.1        # Torque constant [N·m/A]

# PID controller parameters
Kp, Ki, Kd = 8.0, 15.0, 0.2
f_max = 100      # Derivative filter cutoff [Hz]

# Voltage limits
V_min, V_max = -24, 24


# SOURCE SIGNALS ========================================================================

# Speed setpoint: 50 -> 100 -> 75 -> 50 rad/s
spt_amplitudes = [50, 100, 75, 50]
spt_times = [0, 5, 15, 25]

# Load torque: brief spike then sustained load (negative opposes motion)
load_amplitudes = [0, -0.05, 0, -0.02, 0]
load_times = [0, 10, 12, 20, 30]


# SYSTEM SETUP ==========================================================================

# Control blocks
spt = StepSource(amplitude=spt_amplitudes, tau=spt_times)
lod = StepSource(amplitude=load_amplitudes, tau=load_times)
err = Adder("+-")
pid = AntiWindupPID(Kp, Ki, Kd, f_max=f_max, Ks=10, limits=[V_min, V_max])
sat = Clip(min_val=V_min, max_val=V_max)

# Electrical subsystem: L * di/dt = V - R*i - K_e*ω
V_R = Amplifier(-R)         # Voltage drop across resistance
V_L = Amplifier(1/L)        # di/dt calculation
emf = Amplifier(-K_e)       # Back-EMF
V_sum = Adder("+++")        # Voltage summation
I_int = Integrator(0)       # Current integrator

# Mechanical subsystem: J * dω/dt = K_t*i - B*ω - T_load
T_m = Amplifier(K_t)        # Motor torque
T_f = Amplifier(-B)         # Friction torque
T_sum = Adder("+++")        # Torque summation
alp = Amplifier(1/J)        # Angular acceleration
omg = Integrator(0)         # Angular velocity integrator

# Measurement
sco1 = Scope(labels=["Setpoint [rad/s]", "Speed [rad/s]"])
sco2 = Scope(labels=["Current [A]", "Voltage [V]"])

blocks = [
    spt, lod, err, pid, sat,
    V_R, V_L, emf, V_sum, I_int,
    T_m, T_f, T_sum, alp, omg,
    sco1, sco2
]


# CONNECTIONS ===========================================================================

connections = [
    # Control loop
    Connection(spt, err, sco1[0]),
    Connection(omg, err[1], sco1[1]),
    Connection(err, pid),
    Connection(pid, sat),

    # Electrical subsystem
    Connection(sat, V_sum[0], sco2[1]),
    Connection(I_int, V_R),
    Connection(V_R, V_sum[1]),
    Connection(omg, emf),
    Connection(emf, V_sum[2]),
    Connection(V_sum, V_L),
    Connection(V_L, I_int),

    # Mechanical subsystem
    Connection(I_int, T_m, sco2[0]),
    Connection(T_m, T_sum[0]),
    Connection(omg, T_f),
    Connection(T_f, T_sum[1]),
    Connection(lod, T_sum[2]),
    Connection(T_sum, alp),
    Connection(alp, omg)
]


# SIMULATION ============================================================================

Sim = Simulation(blocks, connections, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run simulation
    Sim.run(30)

    # Plot results
    sco1.plot(lw=2)
    sco2.plot(lw=2)

    plt.show()
