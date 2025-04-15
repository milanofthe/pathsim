#########################################################################################
##
##                              Windup of PID Controller
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, Integrator, Amplifier, Adder, Scope, 
    PID, AntiWindupPID, TransferFunction, Delay, Function
    )

from pathsim.solvers import RKCK54


# SYSTEM SETUP AND SIMULATION ===========================================================

#plant gain
K = 0.4

#pid parameters
Kp, Ki, Kd = 1.5, 0.5, 0.2

#source function
def f_s(t):
    if t>100: return 5
    elif t>10: return 10
    else: return 0

#blocks
spt = Source(f_s)  
err = Adder("+-")
# pid = PID(Kp, Ki, Kd, f_max=10)
pid = AntiWindupPID(Kp, Ki, Kd, f_max=10, Ks=10, limits=[-20, 20])
act = Function(lambda x: np.clip(x, -10, 10))
prc = TransferFunction(Residues=[0.1], Poles=[-0.1])
det = Delay(tau=2)
sco = Scope(labels=["s(t)", "x(t)", r"$\epsilon(t)$", "pid(t)"])

blocks = [spt, err, pid, act, prc, det, sco]

connections = [
    Connection(spt, err, sco[0]),
    Connection(det, err[1], sco[1]),
    Connection(err, pid, sco[2]),
    Connection(pid, act, sco[3]),
    Connection(act, prc),
    Connection(prc, det)
]

#simulation initialization
Sim = Simulation(blocks, connections, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(200)

    sco.plot(lw=2)
    plt.show()