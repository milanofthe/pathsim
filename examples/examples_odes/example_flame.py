#########################################################################################
##
##                        PathSim stiff flame ODE example 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Adder, Function
from pathsim.solvers import ESDIRK32, ESDIRK43, GEAR52A


# FLAME INITIAL VALUE PROBLEM ===========================================================

#flame parameter (very stiff)
delta = 0.0001

#blocks that define the system
Int = Integrator(delta)
Fnc = Function(lambda x: x**2 - x**3)

Sco = Scope()

blocks = [Int, Fnc, Sco]

#the connections between the blocks
connections = [
    Connection(Int, Fnc, Sco),
    Connection(Fnc, Int)
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.1, 
    Solver=ESDIRK43, 
    tolerance_lte_abs=1e-6, 
    tolerance_lte_rel=1e-4
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(2/delta)

    #plotting
    Sco.plot(".-")

    plt.show()