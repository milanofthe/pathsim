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
    Connection(Iv, Ix, Fr),# Sc[1]),
    Connection(Ix, Sc[0])
    ]


# event managers ------------------------------------------------------------------------

def func_evt_1(t):
    *_, x = Ix()
    return x

def func_act_1(t):
    *_, x = Ix()
    *_, v = Iv()
    Ix.engine.set(abs(x))
    Iv.engine.set(-b*v)

E1 = ZeroCrossing(
    func_evt=func_evt_1, 
    func_act=func_act_1, 
    tolerance=1e-4
    )


def func_evt_2(t):
    *_, x = Ix()
    return x + 5

def func_act_2(t):
    *_, x = Ix()
    *_, v = Iv()
    Ix.engine.set(abs(x + 5) - 5)
    Iv.engine.set(-b*v)

E2 = ZeroCrossing(
    func_evt=func_evt_2, 
    func_act=func_act_2, 
    tolerance=1e-4
    )

E3 = Condition(
    func_evt=lambda *_: len(E1) >= 10,       # number of events 'E1' (bounces)
    func_act=lambda *_: [E1.off(), E3.off()] # callback switches event tracking
    )


#initialize simulation
Sim = Simulation(
    blocks, 
    connections, 
    events=[E1, E2, E3], 
    dt=dt, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_abs=1e-6, 
    tolerance_lte_rel=1e-4
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation
    Sim.run(15)

    #plot the recordings from the scope
    fig, ax = Sc.plot(lw=2)

    #add detected events to scope plot
    for t in E1: ax.axvline(t, ls="--", c="k")
    for t in E2: ax.axvline(t, ls="-.", c="k")
    for t in E3: ax.axvline(t, ls="-", c="k", lw=2)

    plt.show()