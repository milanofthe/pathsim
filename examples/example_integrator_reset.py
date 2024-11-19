#########################################################################################
##
##                         PathSim event detection example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Scope
from pathsim.events import ZeroCrossing


# INTEGRATOR RESET ======================================================================

#blocks that define the system
Sc = Scope()
I1 = Integrator(0)
Cn = Constant(1)

#blocks of the main system
blocks = [I1, Cn, Sc]

#the connections between the blocks in the main system
connections = [
    Connection(Cn, I1),
    Connection(I1, Sc)
    ]

#events (zero crossings)
E1 = ZeroCrossing(
    blocks=[I1], 
    func_evt=lambda y, x, t: x[0] - 3, 
    func_act=lambda y, x, t: [x[0] - 1]
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=0.06, log=True)

#run simulation for some number of seconds
Sim.run(10)

Sc.plot(".-", lw=1.5)

for e in E1: Sc.ax.axvline(e, ls="--", c="k")

plt.show()