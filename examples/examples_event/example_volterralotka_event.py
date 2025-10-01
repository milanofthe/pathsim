#########################################################################################
##
##                     PathSim Example for Volterra-Lotka System 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, ODE
from pathsim.solvers import RKBS32
from pathsim.events import ZeroCrossingUp, ZeroCrossingDown


# VOLTERRA-LOTKA SYSTEM =================================================================

#simulation timestep
dt = 0.1

#parameters 
alpha = 1.0  # growth rate of prey
beta = 0.1   # predator sucess rate
delta = 0.5  # predator efficiency
gamma = 1.2  # death rate of predators

#function for ODE
def _f(x, u, t):
    x1, x2 = x
    return np.array([x1*(alpha - beta*x2), x2*(delta*x1 - gamma)])

#initial condition
x0 = np.array([5, 10])

#blocks that define the system
VL = ODE(_f, x0)
Sc = Scope(labels=["predators", "prey"])

blocks = [VL, Sc]

#the connections between the blocks
connections = [
    Connection(VL, Sc),
    Connection(VL[1], Sc[1]),
    ]


#events to detect
def func_evt_1(t):
    i, o, s = VL()
    return s[0] - 4

def func_evt_2(t):
    i, o, s = VL()
    return s[1] - 4

E1 = ZeroCrossingUp(
    func_evt=func_evt_1,
    tolerance=1e-4
    )

E2 = ZeroCrossingUp(
    func_evt=func_evt_2,
    func_act=lambda _: Sim.stop(),
    tolerance=1e-4
    )

E3 = ZeroCrossingDown(
    func_evt=func_evt_1,
    tolerance=1e-4
    )

E4 = ZeroCrossingDown(
    func_evt=func_evt_2,
    tolerance=1e-4
    )

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    [E1, E2, E3, E4],
    dt=dt, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_rel=1e-4, 
    tolerance_lte_abs=1e-6
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation
    Sim.run(10)

    fig, ax = Sc.plot(".-")

    #add detected events to scope plot
    for e in E1: ax.axvline(e, ls="--", c="k")
    for e in E2: ax.axvline(e, ls=":", c="k")
    for e in E3: ax.axvline(e, ls="--", c="k")
    for e in E4: ax.axvline(e, ls=":", c="k")


    plt.show()

        