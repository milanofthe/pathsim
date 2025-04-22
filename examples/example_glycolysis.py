#########################################################################################
##
##                      Pathsim Example of Glycolysis reaction
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import RKCK54


# MODEL =================================================================================

a, b = 0.08, 0.6

xy_0 = np.zeros(2)

def f_gly(_x, u, t):
    x, y = _x
    dxdt = -x + a * y + x**2 * y
    dydt = b - a * y - x**2 * y
    return np.array([dxdt, dydt])

gly = ODE(func=f_gly, initial_value=xy_0)
sco = Scope(labels=["x", "y"])

blocks = [gly, sco]

#connections between the blocks
connections = [
    Connection(gly[:2], sco[:2])
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