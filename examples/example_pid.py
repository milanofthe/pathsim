#########################################################################################
##
##                         PathSim Example for PID-controller 
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope, Differentiator

from pathsim.solvers import RKCK54, GEAR52A, ESDIRK43


# SYSTEM SETUP AND SIMULATION ===========================================================

# System parameters
K = 0.2    # System gain
T = 2.0    # System time constant
Kp = 1.5   # Proportional gain
Ki = 0.75  # Integral gain
Kd = 0.05  # Derivative gain

# Blocks
setpoint = Source(lambda t: int(t>10)-0.5*int(t>50))  # Step inputs 
error = Adder()
amp_P = Amplifier(Kp)
amp_I = Amplifier(Ki)
amp_D = Amplifier(Kd)
I = Integrator()
D = Differentiator(f_max=10) 
pid_sum = Adder()
plant = Integrator()
plant_gain = Amplifier(K)
feedback = Amplifier(-1)
scope = Scope(labels=["setpoint", "output", "control signal"])

#Blocks 
blocks = [setpoint, error, amp_P, amp_I, amp_D, I, D, pid_sum, plant, plant_gain, feedback, scope]

# Connections
connections = [
    Connection(setpoint, error, scope[0]),
    Connection(feedback, error[1]),
    Connection(error, amp_P, amp_I, amp_D),
    Connection(amp_I, I),
    Connection(amp_D, D),
    Connection(amp_P, pid_sum[0]),
    Connection(I, pid_sum[1]),
    Connection(D, pid_sum[2]),
    Connection(pid_sum, plant, scope[2]),
    Connection(plant, plant_gain),
    Connection(plant_gain, feedback, scope[1])
]

# Simulation initialization
sim = Simulation(blocks, connections, dt=0.1, Solver=RKCK54)

#run the simulation for some time
sim.run(100)

scope.plot()

plt.show()