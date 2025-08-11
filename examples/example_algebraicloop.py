#########################################################################################
##
##                         PathSim example of an algebraic loop
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, 
    Amplifier, 
    Adder, 
    Scope, 
    )


# ALGEBRAIC LOOP ========================================================================

#simulation timestep
dt = 0.1

#algebraic feedback
a = -0.2

#blocks that define the system
Src = Source(lambda t: 2*np.cos(t))
Amp = Amplifier(a)
Add = Adder()
Sco = Scope(labels=["src", "amp", "add"])

blocks = [Src, Amp, Add, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Add),
    Connection(Add, Amp),
    Connection(Amp, Add[1]),
    Connection(Src, Sco),
    Connection(Amp, Sco[1]),
    Connection(Add, Sco[2])
    ]


#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(5)

    Sco.plot(".-")

    plt.show()