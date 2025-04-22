#########################################################################################
##
##      Pathsim Example of Nerve impulse action potential (Bonhoeffer-Van der Pol)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Source, Scope
from pathsim.solvers import RKCK54


# MODEL =================================================================================

a, b, c, F, omega = 0.7, 0.8, 0.1, 0.6, 1

xy_0 = np.array([0.0, 0.0])

def f_bhf(_x, u, t):
    x, y = _x
    dxdt = x - x**3 / 3 - y + u[0]
    dydt = c * (x + a - b * y)
    return np.array([dxdt, dydt])

def s_bhf(t):
    return F * np.cos(omega * t)

src = Source(s_bhf)
bhf = ODE(func=f_bhf, initial_value=xy_0)
sco = Scope(labels=["x", "y"])

blocks = [src, bhf, sco]

#connections between the blocks
connections = [
    Connection(bhf[:2], sco[:2]),
    Connection(src, bhf)
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

    Sim.run(100)

    sco.plot()

    sco.plot2D()

    plt.show()