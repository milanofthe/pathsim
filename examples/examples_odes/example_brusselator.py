#########################################################################################
##
##                          PathSim Example of Brusselator ODE
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import SSPRK33, RKCK54


# MODEL =================================================================================

a, b = 0.4, 1.2
# a, b = 1.0, 1.7

def f_bru(_x, u, t):
    x, y = _x
    dxdt = a - x - b * x + x**2 * y
    dydt = b * x - x**2 * y
    return np.array([dxdt, dydt])

bru = ODE(func=f_bru, initial_value=np.zeros(2))
sco = Scope(labels=["x", "y"])

blocks = [bru, sco]

#connections between the blocks
connections = [
    Connection(bru[:2], sco[:2])
    ]

#create a simulation instance from the blocks and connections
Sim = Simulation(
    blocks, 
    connections, 
    Solver=RKCK54,
    tolerance_lte_rel=1e-6,
    tolerance_lte_abs=1e-9
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(150)

    sco.plot()
    sco.plot2D()

    plt.show()