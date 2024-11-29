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

def func_evt(blocks, t):
    b = blocks[0]
    i, o, s = b()
    return s - 3

def func_act(blocks, t):
    b = blocks[0]
    i, o, s = b()
    b.engine.set(s - 1)

E1 = ZeroCrossing(
    blocks=[I1], 
    func_evt=func_evt, 
    func_act=func_act
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=0.06, log=True)

#run simulation for some number of seconds
Sim.run(10)

Sc.plot(".-", lw=1.5)

for e in E1: Sc.ax.axvline(e, ls="--", c="k")

plt.show()