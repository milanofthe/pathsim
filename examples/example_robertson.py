#########################################################################################
##
##                        PathSim Robertson ODE Example 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ODE
from pathsim.solvers import GEAR52A


# ROBERTSON ODE INITIAL VALUE PROBLEM ===================================================

#initial condition
x0 = np.array([1, 0, 0])

def func(x, u, t):
    return np.array([
        -0.04*x[0] + 1e4*x[1]*x[2],
        0.04*x[0] - 1e4*x[1]*x[2] - 3e7*x[1]**2,
        3e7*x[1]**2
    ])

#blocks that define the system
VDP = ODE(func, x0)
Sco = Scope(labels=["x", "y", "z"])

blocks = [VDP, Sco]

#the connections between the blocks
connections = [
    Connection(VDP, Sco),
    Connection(VDP[1], Sco[1]),
    Connection(VDP[2], Sco[2])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.0001, 
    log=True, 
    Solver=GEAR52A, 
    tolerance_lte_abs=1e-8, 
    tolerance_lte_rel=1e-6
    )

Sim.run(3)

#plotting
Sco.plot(".-")

plt.show()