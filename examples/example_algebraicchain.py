#########################################################################################
##
##                 PathSim example of a purely algebraic signal chain
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, 
    Constant,
    Function, 
    Amplifier, 
    Adder, 
    Scope
    )

# Algebraic Signal ======================================================================

#blocks that define the system
Src = Source(np.sin)
Cns = Constant(-1/2)
Amp = Amplifier(2)
Fnc = Function(lambda x: x**2)
Add = Adder()
Sc1 = Scope(labels=["sin"])
Sc2 = Scope(labels=["a", "b"])

blocks = [Src, Cns, Amp, Fnc, Add, Sc1, Sc2]

#the connections between the blocks
connections = [
    Connection(Src, Fnc, Sc1),
    Connection(Fnc, Add[0], Sc2[0]),
    Connection(Cns, Add[1]),
    Connection(Add, Amp),
    Connection(Amp, Sc2[1])
    ]

#initialize simulation with the blocks, connections
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.01
    )

# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(12)

    Sc1.plot(lw=2)
    Sc2.plot(lw=2)

    plt.show()