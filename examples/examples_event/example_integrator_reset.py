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

def func_evt(t):
    i, o, s = I1()
    return s - 3

def func_act(t):
    i, o, s = I1()
    I1.engine.set(s - 1)

E1 = ZeroCrossing(
    func_evt=func_evt, 
    func_act=func_act
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=0.06, log=True)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run simulation for some number of seconds
    Sim.run(10)

    fig, ax = Sc.plot(".-", lw=1.5)

    for e in E1: ax.axvline(e, ls="--", c="k")

    plt.show()