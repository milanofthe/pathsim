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
from pathsim.solvers import GEAR52A, ESDIRK43


# ROBERTSON ODE INITIAL VALUE PROBLEM ===================================================

#parameters
a, b, c = 0.04, 1e4, 3e7

#initial condition
x0 = np.array([1, 0, 0])

def func(x, u, t):
    return np.array([
        -a*x[0] + b*x[1]*x[2],
         a*x[0] - b*x[1]*x[2] - c*x[1]**2,
                                c*x[1]**2
    ])

#blocks that define the system
Rob = ODE(func, x0)
Sco = Scope(labels=["x", "y", "z"])

blocks = [Rob, Sco]

#the connections between the blocks
connections = [
    Connection(Rob[0], Sco[0]),
    Connection(Rob[1], Sco[1]),
    Connection(Rob[2], Sco[2])
    ]

#initialize simulation with the blocks, connections, timestep
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.0001, 
    log=True, 
    Solver=GEAR52A, 
    tolerance_lte_abs=1e-6, 
    tolerance_lte_rel=1e-4,
    tolerance_fpi=1e-9
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(100)

    #plotting
    Sco.plot(".-")

    plt.show()