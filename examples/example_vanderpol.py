#########################################################################################
##
##                        PathSim Van der Pol System Example 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ODE
from pathsim.solvers import ESDIRK32, ESDIRK54


# VAN DER POL OSCILLATOR INITIAL VALUE PROBLEM ==========================================

#initial condition
x0 = np.array([2, 0])

#van der Pol parameter
mu = 1000

def func(x, u, t):
    return np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])

#analytical jacobian (optional)
def jac(x, u, t):
    return np.array([[0, 1], [-mu*2*x[0]*x[1]-1, mu*(1 - x[0]**2)]])

#blocks that define the system
VDP = ODE(func, x0, jac) #jacobian improves convergence but is not needed
Sco = Scope()

blocks = [VDP, Sco]

#the connections between the blocks
connections = [
    Connection(VDP, Sco),
    Connection(VDP[1], Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=0.5, log=True, Solver=ESDIRK54, tolerance_lte_abs=1e-6, tolerance_lte_rel=1e-4)

Sim.run(3*mu)

#plotting
Sco.plot(".-")

plt.show()