#########################################################################################
##
##                   PathSim Example for stick-slip effect modeling
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, ODE, Scope
from pathsim.solvers import RKCK54, ESDIRK43


# SIMULATING THE STICK-SLIP EFFECT ======================================================

#simulation and model parameters
m = 20.0    # mass
k = 70.0    # spring constant
d = 10.0    # spring damping
mu = 1.5    # friction coefficient
g = 9.81    # gravity
v = 3.0     # belt velocity magnitude
T = 50.0    # excitation period

#ODE function
def _f(x, u, t):
    v_rel = x[1] - u[0] #relative velocity
    F_c = mu*m*g*np.tanh(1000*v_rel) #coulomb friction force magnitude
    return np.array([ x[1], -(k*x[0] + d*x[1] + F_c)/m])

#blocks that define the system
Sr = Source(lambda t: v*np.sin(2*np.pi*t/T)) #this is the velocity of the belt
St = ODE(_f, np.zeros(2))
Sc = Scope(labels=["belt velocity", "box position", "box velocity"])

blocks = [Sr, St, Sc]

#the connections between the blocks
connections = [
    Connection(Sr, St, Sc),
    Connection(St, Sc[1]),
    Connection(St[1], Sc[2])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=0.01, log=True, Solver=ESDIRK43, tolerance_lte_abs=1e-5, tolerance_lte_rel=1e-4)

#run the simulation for some time
Sim.run(2*T)

#plot the result directly from the scope
Sc.plot()

plt.show()