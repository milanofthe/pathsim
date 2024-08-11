#########################################################################################
##
##                     PathSim Example for Volterra-Lotka System 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ODE
from pathsim.solvers import RKCK54, RKF78



# VOLTERRA-LOTKA SYSTEM =================================================================

#simulation timestep
dt = 0.1

#parameters 
alpha = 1.0  # growth rate of prey
beta = 0.1   # predator sucess rate
delta = 0.5  # predator efficiency
gamma = 1.2  # death rate of predators

#functon for ODE
def _f(x, u, t):
    x1, x2 = x
    return np.array([x1*(alpha - beta*x2), x2*(delta*x1 - gamma)])


fig, ax = plt.subplots(tight_layout=True)
ax.set_xlabel("predator population")
ax.set_ylabel("prey population")

#iterate different initial conditions for predators
for r in np.linspace(3, 12, 10):

    #initial condition
    x0 = np.array([r, 10])

    #blocks that define the system
    VL = ODE(_f, x0)
    Sc = Scope(labels=["predators", "prey"])

    blocks = [VL, Sc]

    #the connections between the blocks
    connections = [
        Connection(VL, Sc),
        Connection(VL[1], Sc[1]),
        ]

    #initialize simulation with the blocks, connections, timestep and logging enabled
    Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=RKCK54)
        
    #run the simulation
    Sim.run(50)

    #read the data from the scope
    time, data = Sc.read()

    #plot the phase diagram
    ax.plot(*data, label=f"{x0}")

ax.legend()

plt.show()

    