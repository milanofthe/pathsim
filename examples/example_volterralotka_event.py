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
from pathsim.solvers import RKBS32
from pathsim.events import ZeroCrossing


# VOLTERRA-LOTKA SYSTEM =================================================================

#simulation timestep
dt = 0.1

#parameters 
alpha = 1.0  # growth rate of prey
beta = 0.1   # predator sucess rate
delta = 0.5  # predator efficiency
gamma = 1.2  # death rate of predators

#function for ODE
def _f(x, u, t):
    x1, x2 = x
    return np.array([x1*(alpha - beta*x2), x2*(delta*x1 - gamma)])

#initial condition
x0 = np.array([5, 10])

#blocks that define the system
VL = ODE(_f, x0)
Sc = Scope(labels=["predators", "prey"])

blocks = [VL, Sc]

#the connections between the blocks
connections = [
    Connection(VL, Sc),
    Connection(VL[1], Sc[1]),
    ]

#events to detect
E1 = ZeroCrossing(
    blocks=[VL],
    g=lambda x: x[1] - 4,
    direction=-1,
    tolerance=1e-4
    )

E2 = ZeroCrossing(
    blocks=[VL],
    g=lambda x: x[0] - 4,
    direction=-1,
    tolerance=1e-4
    )

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    [E1, E2],
    dt=dt, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_rel=1e-4, 
    tolerance_lte_abs=1e-6
    )

#run the simulation
Sim.run(20)

Sc.plot(".-")

#add detected events to scope plot
for e in E1: Sc.ax.axvline(e, ls="--", c="k")
for e in E2: Sc.ax.axvline(e, ls=":", c="k")

#read the data from the scope
time, data = Sc.read()

#plot the phase diagram
fig, ax = plt.subplots(tight_layout=True)
ax.plot(*data, label=f"{x0}")
ax.set_xlabel("predator population")
ax.set_ylabel("prey population")
ax.legend()






plt.show()

    