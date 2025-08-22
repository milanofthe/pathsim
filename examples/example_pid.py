#########################################################################################
##
##                         PathSim Example for PID-controller 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope, PID
from pathsim.solvers import RKCK54, RKBS32
from pathsim.optim import Value


# SYSTEM SETUP AND SIMULATION ===========================================================

#plant gain
K = 0.4

#pid parameters
Kp, Ki, Kd = Value.array([1.5, 0.5, 0.1]) 

#source function
def f_s(t):
    if t>60: return 0.5
    elif t>20: return 1
    else: return 0 

#blocks
spt = Source(f_s)  
err = Adder("+-")
pid = PID(Kp, Ki, Kd, f_max=10)
pnt = Integrator()
pgn = Amplifier(K)
sco = Scope(labels=["s(t)", "x(t)", r"$\epsilon(t)$"])

blocks = [spt, err, pid, pnt, pgn, sco]

connections = [
    Connection(spt, err, sco[0]),
    Connection(pgn, err[1], sco[1]),
    Connection(err, pid, sco[2]),
    Connection(pid, pnt),
    Connection(pnt, pgn)
]

#simulation initialization
Sim = Simulation(blocks, connections, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(100)

    sco.plot(".-", lw=2)

    #plot sensitivities
    time, [sp, ot, er] = sco.read()

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120, tight_layout=True)

    ax.plot(time, Value.der(er, Kp), lw=2, label=r"$\partial \epsilon / \partial K_p $")
    ax.plot(time, Value.der(er, Ki), lw=2, label=r"$\partial \epsilon / \partial K_i $")
    ax.plot(time, Value.der(er, Kd), lw=2, label=r"$\partial \epsilon / \partial K_d $")

    ax.legend(fancybox=False)
    ax.set_xlabel("time [s]")
    ax.grid()

    plt.show()