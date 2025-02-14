#########################################################################################
##
##                        PathSim Example for TransferFunction
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Scope, TransferFunction
from pathsim.solvers import RKCK54, GEAR52A


# STEP RESPONSE OF A TRANSFER FUNCTION ==================================================

#step delay
tau = 5.0

#simulation timestep
dt = 0.05

#transfer function parameters
const    = 0.0
poles    = [-0.3, -0.05+0.4j, -0.05-0.4j, -0.1+2j, -0.1-2j]
residues = [-0.2,      -0.2j,       0.2j,     0.3,     0.3]

#blocks and connections
Sr = Source(lambda t: int(t>=tau))
TF = TransferFunction(Poles=poles, Residues=residues, Const=const)
Sc = Scope(labels=["step", "response"])

blocks = [Sr, TF, Sc]

connections = [
    Connection(Sr, TF, Sc), 
    Connection(TF, Sc[1]) 
    ]

#initialize simulation
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=RKCK54)

#run simulation
Sim.run(100)

#plot the results from the scope directly
Sc.plot()


plt.show()