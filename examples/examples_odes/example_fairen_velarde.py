#########################################################################################
##
##           PathSim Example of Bacterial respiration by Fairen and Velarde
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import SSPRK33, RKCK54


# MODEL =================================================================================

A, B, Q = 2.0, 3.0, 6.5
# A, B, Q = 2.0, 3.0, 3.5

def f_bac(_x, u, t):
    x, y = _x
    dxdt = B - x - x * y / (1 + Q * x**2)
    dydt = A - x * y / (1 + Q * x**2)
    return np.array([dxdt, dydt])

bac = ODE(func=f_bac, initial_value=np.zeros(2))
sco = Scope(labels=["x", "y"])

blocks = [bac, sco]

#connections between the blocks
connections = [
    Connection(bac[:2], sco[:2])
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

    #plot the results directly from the scope
    sco.plot()
    sco.plot2D()

    plt.show()