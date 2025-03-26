#########################################################################################
##
##           example of a simple feedback system with steady state analysis
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

from pathsim.solvers import RKBS32


# 1st ORDER SYSTEM ======================================================================

#simulation timestep
dt = 0.01

#step delay
tau = 1.5

#blocks that define the system
Src = Source(lambda t: int(t>tau))
Int = Integrator()
Amp = Amplifier(-0.5)
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
Sim = Simulation(blocks, connections, dt=dt, log=True)
    

# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(2*tau)

    #then force to steady state
    Sim.steadystate(reset=False)

    #then run some more 
    Sim.run(tau, reset=False)

    Sco.plot()

    plt.show()