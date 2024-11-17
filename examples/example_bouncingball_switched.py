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

from pathsim.events import ZeroCrossing, Condition


# BOUNCING BALL SYSTEM ==================================================================

#simulation timestep
dt = 0.01

#gravitational acceleration
g = 9.81

#elasticity of bounce
b = 0.95

#mass normalized friction coefficientand
k = 0.2

#initial values
x0, v0 = 1, 10

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
    Connection(Iv, Ix, Fr, Sc[1]),
    Connection(Ix, Sc[0])
    ]


#events (zero crossing)
E1 = ZeroCrossing(
    blocks=[Ix, Iv],               # blocks to watch states of
    g=lambda x, y: x,              # event function for zero crossing detection
    f=lambda x, y: [abs(x), -b*y], # action function for state transformation
    tolerance=1e-4
    )

E2 = ZeroCrossing(
    blocks=[Ix, Iv],                       # blocks to watch states of
    g=lambda x, y: x + 3,                  # event function for zero crossing detection
    f=lambda x, y: [abs(x + 3) - 3, -b*y], # action function for state transformation
    tolerance=1e-4
    )

E3 = Condition(
    g=lambda: len(E1) >= 13,       # number of events 'E1' (bounces)
    h=lambda: [E1.off(), E3.off()] # callback switches event tracking
    )

events = [E1, E2, E3]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    dt=dt, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_rel=1e-3, 
    tolerance_lte_abs=1e-5
    )

#run the simulation
Sim.run(15)


#plot the recordings from the scope
Sc.plot(".-", lw=2)

#add detected events to scope plot
for t in E1: Sc.ax.axvline(t, ls="--", c="k")
for t in E2: Sc.ax.axvline(t, ls="-.", c="k")
for t in E3: Sc.ax.axvline(t, ls="-", c="k", lw=3)

plt.show()