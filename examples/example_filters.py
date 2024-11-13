#########################################################################################
##
##                    PathSim Example for Filters from RF Toolbox
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Scope
from pathsim.blocks.rf import SquareWaveSource, ButterworthLowpassFilter
from pathsim.solvers import SSPRK33


# FILTERING A SQUAREWAVE ================================================================

dt = 0.02

#filter bandwidth, order and signal frequency
B, n, f = 2, 6, 1

#blocks that define the system
Src = SquareWaveSource(frequency=f)
LPF = ButterworthLowpassFilter(B, n)
Sco = Scope(labels=["source", "output"])

blocks = [Src, LPF, Sco]

#the connections between the blocks
connections = [
    Connection(Src, LPF, Sco),
    Connection(LPF, Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=SSPRK33)

#run the simulation 
Sim.run(10)

#plot the results from the scope directly
Sco.plot()

plt.show()