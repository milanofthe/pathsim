#########################################################################################
##
##                      PathSim Example of Differentiable Simulation
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

#optimization module
from pathsim.diff import Parameter, Value


# 1st ORDER FEEDBACK SYSTEM =============================================================

#parameters
a = Parameter(-1)
z = Parameter(2)
b = Parameter(1)

#simulation timestep
dt = 0.01

#step delay
tau = 3

#blocks that define the system
Src = Source(lambda t: b*int(t>tau))
Int = Integrator(z)
Amp = Amplifier(a)
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
    
#run the simulation for some time
Sim.run(4*tau)

Sco.plot()


#plot derivatives -----------------------------------------------------------------------

time, [_, res] = Sco.read()

fig, ax = plt.subplots(nrows=1, figsize=(8, 4), tight_layout=True, dpi=120)

ax.plot(time, list(map(lambda x:x.d(a), res)), label="$dx/da$")
ax.plot(time, list(map(lambda x:x.d(b), res)), label="$dx/db$")
ax.plot(time, list(map(lambda x:x.d(z), res)), label="$dx/dx_0$")

ax.set_xlabel("time [s]")

ax.grid(True)
ax.legend()

plt.show()