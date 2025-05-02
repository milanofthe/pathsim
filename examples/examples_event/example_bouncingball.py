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
from pathsim.solvers import RKF21

from pathsim.events import ZeroCrossing


# BOUNCING BALL SYSTEM ==================================================================

#simulation timestep
dt = 0.01

#gravitational acceleration
g = 9.81

#elasticity of bounce
b = 0.95

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
def func_evt(t):
    *_, x = Ix() #get block outputs and states
    return x

#action function for state transformation
def func_act(t):
    *_, x = Ix()
    *_, v = Iv()
    Ix.engine.set(abs(x))
    Iv.engine.set(-b*v)

#events (zero crossing)
E1 = ZeroCrossing(
    func_evt=func_evt,                 
    func_act=func_act, 
    tolerance=1e-6
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    Solver=RKF21, 
    tolerance_lte_rel=1e-4, 
    tolerance_lte_abs=1e-7
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation
    Sim.run(20)

    #read the recordings from the scope
    time, [x] = Sc.read()

    #plot the recordings from the scope
    fig, ax = Sc.plot(".-", lw=2)

    #add detected events to scope plot
    for t in E1: ax.axvline(t, ls="--", c="k")


    # timesteps -----------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8,4), tight_layout=True, dpi=120)

    for t in E1: ax.axvline(t, ls="--", c="k")

    ax.plot(time[:-1], np.diff(time), lw=2)

    ax.set_yscale("log")
    ax.set_ylabel("dt [s]")
    ax.set_xlabel("time [s]")
    ax.grid(True)


    plt.show()