#########################################################################################
##
##              PathSim Example of Thomas cyclically symmetric attractor
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import SSPRK33, RKCK54


# MODEL =================================================================================

b = 0.22
xyz_0 = np.array([1, 1, -3])

def f_tcs(_x, u, t):
    x, y, z = _x
    dxdt = np.sin(y) - b*x
    dydt = np.sin(z) - b*y
    dzdt = np.sin(x) - b*z
    return np.array([dxdt, dydt, dzdt])

tcs = ODE(func=f_tcs, initial_value=xyz_0)
sco = Scope(labels=["x", "y", "z"])

blocks = [tcs, sco]

#connections between the blocks
connections = [
    Connection(tcs[:3], sco[:3])
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

    Sim.run(500)

    sco.plot3D()

    plt.show()