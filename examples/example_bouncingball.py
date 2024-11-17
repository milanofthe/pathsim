#########################################################################################
##
##                    PathSim example of event detection mechanism
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Function, Adder, Scope
from pathsim.solvers import RKBS32, SSPRK33

from pathsim.events import ZeroCrossing
from pathsim.diff import Value, der


# BOUNCING BALL SYSTEM ==================================================================

#simulation timestep
dt = 0.01

#gravitational acceleration
g = Value(9.81)

#elasticity of bounce
b = Value(0.9)

#mass normalized friction coefficientand
k = Value(0.4)

#initial values
x0, v0 = Value(1), Value(10)

#newton friction
def fric(v): 
    return -k * np.sign(v) * v**2

#blocks that define the system
Ix = Integrator(x0)     # v -> x
Iv = Integrator(v0)     # a -> v 
Fr = Function(fric)     # newton friction
Ad = Adder()
Cn = Constant(-g)       # gravitational acceleration
Sc = Scope(labels=["x", "v"])

blocks = [Ix, Iv, Fr, Ad, Cn, Sc]

#the connections between the blocks
connections = [
    Connection(Cn, Ad[0]),
    Connection(Fr, Ad[1]),
    Connection(Ad, Iv),
    Connection(Iv, Ix, Fr),
    Connection(Ix, Sc[0])
    ]

#events (zero crossing)
E1 = ZeroCrossing(
    blocks=[Ix, Iv],               # blocks to watch states of
    g=lambda x, y: x,              # event function for zero crossing detection
    f=lambda x, y: [abs(x), -b*y], # action function for state transformation
    tolerance=1e-3
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    dt=dt, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_rel=0.0, 
    tolerance_lte_abs=1e-4
    )

#run the simulation
Sim.run(5)

#read the recordings from the scope
time, [x] = Sc.read()

#plot the recordings from the scope
Sc.plot(".-", lw=2)

#add detected events to scope plot
for t in E1: Sc.ax.axvline(t, ls="--", c="k")


# timesteps -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8,4), tight_layout=True, dpi=120)

for t in E1: ax.axvline(t, ls="--", c="k")

ax.plot(time[:-1], np.diff(time), lw=2)

ax.set_yscale("log")
ax.set_ylabel("dt [s]")
ax.set_xlabel("time [s]")
ax.grid(True)


# derivatives ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8,4), tight_layout=True, dpi=120)

for t in E1: ax.axvline(t, ls="--", c="k")

ax.plot(time, der(x, k), lw=2, label="$dx/dk$")
ax.plot(time, der(x, g), lw=2, label="$dx/dg$")
ax.plot(time, der(x, b), lw=2, label="$dx/db$")
ax.plot(time, der(x, v0), lw=2, label="$dx/dv_0$")
ax.plot(time, der(x, x0), lw=2, label="$dx/dx_0$")

ax.set_xlabel("time [s]")
ax.legend()
ax.grid(True)

plt.show()