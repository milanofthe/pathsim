#########################################################################################
##
##                  PathSim event detection example with thermostat
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Scope, Amplifier, Adder
from pathsim.solvers import RKBS32
from pathsim.events import ZeroCrossingUp, ZeroCrossingDown


# THERMOSTAT ============================================================================


#system parameters
a = 0.45
T = 15
H = 5
Kp = 25
Km = 23

#blocks that define the system
sco = Scope(labels=["temperature", "heater"])
integ = Integrator(T)
feedback = Amplifier(-a)
heater = Constant(H)
ambient = Constant(a*T)
add = Adder()

#blocks of the main system
blocks = [sco, integ, feedback, heater, ambient, add]

#the connections between the blocks in the main system
connections = [
    Connection(integ, feedback, sco),
    Connection(feedback, add),
    Connection(heater, add[1], sco[1]),
    Connection(ambient, add[2]),
    Connection(add, integ)
    ]

#events (zero crossings)

def func_evt_up(t):
    *_, x = integ()
    return x - Kp

def func_act_up(t):
    heater.off()

E1 = ZeroCrossingUp(
    func_evt=func_evt_up, 
    func_act=func_act_up
    )


def func_act_down(t):
    heater.on()
 
def func_evt_down(t):
    *_, x = integ()
    return x - Km

E2 = ZeroCrossingDown(
    func_evt=func_evt_down, 
    func_act=func_act_down
    )

events = [E1, E2]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    dt=0.1, 
    dt_max=0.05, 
    log=True, 
    Solver=RKBS32
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run simulation for some number of seconds
    Sim.run(30)

    fig, ax = sco.plot(lw=2)

    # #thermostat switching events
    for e in E1: ax.axvline(e, ls="--", c="k")
    for e in E2: ax.axvline(e, ls=":", c="k")

    plt.show()