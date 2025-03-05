#########################################################################################
##
##                      PathSim Example of Differentiable Simulation
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

#optimization module
from pathsim.optim import Value, der

from pathsim.solvers import RKBS32


# 1st ORDER FEEDBACK SYSTEM =============================================================

#parameters
a = Value(-1)
z = Value(2)
b = Value(1)

#simulation timestep
dt = 0.02

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
Sim = Simulation(
    blocks, 
    connections, 
    dt=dt, 
    log=True,
    Solver=RKBS32
    )
    
#run the simulation for some time
Sim.run(4*tau)

Sco.plot(".-")


#plot derivatives -----------------------------------------------------------------------

time, [_, res] = Sco.read()

fig, ax = plt.subplots(nrows=1, figsize=(8, 4), tight_layout=True, dpi=120)

ax.plot(time, der(res, a), label="$dx/da$")
ax.plot(time, der(res, b), label="$dx/db$")
ax.plot(time, der(res, z), label="$dx/dx_0$")

ax.set_xlabel("time [s]")

ax.grid(True)
ax.legend()

plt.show()