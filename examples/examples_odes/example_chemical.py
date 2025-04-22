#########################################################################################
##
##     Example of Chemical reaction (chlorine dioxide–iodine–malonic acid reaction)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import RKCK54


# MODEL =================================================================================

a, b = 10, 1

xy_0 = np.zeros(2)

def f_che(_x, u, t):
    x, y = _x
    dxdt = a - x - 4 * x * y / (1 + x**2)
    dydt = b * x * (1 - y / (1 + x**2))
    return np.array([dxdt, dydt])

che = ODE(func=f_che, initial_value=xy_0)
sco = Scope(labels=["x", "y"])

blocks = [che, sco]

#connections between the blocks
connections = [
    Connection(che[:2], sco[:2])
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

    Sim.run(50)

    sco.plot()

    sco.plot2D()

    plt.show()