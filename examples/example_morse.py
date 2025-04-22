#########################################################################################
##
##                          Pathsim Example of morse equation
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Source, Scope
from pathsim.solvers import RKCK54


# MODEL =================================================================================

alpha, beta, F, omega = 0.8, 8, 2.5, 4.171

xy_0 = np.array([2, 3])

def f_mrs(x, u, t):
    x0, x1 = x
    dx0dt = x1
    dx1dt = -alpha * x1 - beta * (1 - np.exp(-x0)) * np.exp(-x0) + u[0]
    return np.array([dx0dt, dx1dt])

def s_mrs(t):
    return F * np.cos(omega * t)

src = Source(s_mrs)
mrs = ODE(func=f_mrs, initial_value=xy_0)
sco = Scope(labels=["x", "y"])

blocks = [src, mrs, sco]

#connections between the blocks
connections = [
    Connection(mrs[:2], sco[:2]),
    Connection(src, mrs)
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