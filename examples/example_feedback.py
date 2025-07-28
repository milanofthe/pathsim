#########################################################################################
##
##                    PathSim example of a simple feedback system
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, 
    Integrator, 
    Amplifier, 
    Adder, 
    Scope
    )


# 1st ORDER SYSTEM ======================================================================

#step delay
tau = 3

#blocks that define the system
Src = Source(lambda t : int(t>tau) )
Int = Integrator(2)
Amp = Amplifier(-1)
Add = Adder()
Sco = Scope(labels=["step", "response"])

blocks = [Src, Int, Amp, Add, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Add[0], Sco[0]),
    Connection(Amp, Add[1]),
    Connection(Add, Int),
    Connection(Int, Amp, Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.01
    )

# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(4*tau)

    Sco.plot(lw=2)

    plt.show()