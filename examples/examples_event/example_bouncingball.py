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
from pathsim.solvers import RKBS32

from pathsim.events import ZeroCrossing


# BOUNCING BALL SYSTEM ==================================================================

#simulation timestep
dt = 0.01

#gravitational acceleration
g = 9.81

#elasticity of bounce
b = 0.9

#initial values
x0, v0 = 1, 5

#blocks that define the system
Ix = Integrator(x0)     # v -> x
Iv = Integrator(v0)     # a -> v 
Cn = Constant(-g)       # gravitational acceleration
Sc = Scope(labels=["x", "v"])

blocks = [Ix, Iv, Cn, Sc]

#the connections between the blocks
connections = [
    Connection(Cn, Iv),
    Connection(Iv, Ix),
    Connection(Ix, Sc[0])
    ]

#event function for zero crossing detection
def func_evt(blocks, t):
    b1, b2 = blocks
    *_, s = b1() #get block outputs and states
    return s

#action function for state transformation
def func_act(blocks, t):
    b1, b2 = blocks
    *_, s1 = b1()
    *_, s2 = b2()
    b1.engine.set(abs(s1))
    b2.engine.set(-b*s2)

#events (zero crossing)
E1 = ZeroCrossing(
    blocks=[Ix, Iv],    # blocks to watch 
    func_evt=func_evt,                 
    func_act=func_act, 
    tolerance=1e-4
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    dt=dt, 
    dt_max=5*dt,
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_rel=1e-5, 
    tolerance_lte_abs=1e-7
    )

#run the simulation
Sim.run(10)

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


plt.show()