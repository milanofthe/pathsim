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
from pathsim.solvers import RKCK54
from pathsim.optim import Value


# SYSTEM SETUP AND SIMULATION ===========================================================

#system parameters
K = 0.3                    # System gain
Kp = Value(3, sig=2)       # Proportional gain
Ki = Value(2, sig=1)       # Integral gain
Kd = Value(0.1, sig=0.05)  # Derivative gain

#blocks
spt = Source(lambda t: int(t>10)-0.5*int(t>50))  # Step inputs 
err = Adder("+-")
pid = PID(Kp, Ki, Kd, f_max=10)
pnt = Integrator()
pgn = Amplifier(K)
sco = Scope(labels=["setpoint", "output", "control signal", "error"])

blocks = [spt, err, pid, pnt, pgn, sco]

connections = [
    Connection(spt, err, sco[0]),
    Connection(pgn, err[1], sco[1]),
    Connection(err, pid, sco[3]),
    Connection(pid, pnt, sco[2]),
    Connection(pnt, pgn)
]

#simulation initialization
Sim = Simulation(blocks, connections, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(100)

    sco.plot()

    #plot sensitivities
    time, [sp, ot, ct, er] = sco.read()

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120, tight_layout=True)

    ax.plot(time, Value.der(er, Kp), label=r"$\partial \epsilon / \partial K_p $")
    ax.plot(time, Value.der(er, Ki), label=r"$\partial \epsilon / \partial K_i $")
    ax.plot(time, Value.der(er, Kd), label=r"$\partial \epsilon / \partial K_d $")

    ax.legend(fancybox=False)
    ax.set_xlabel("time [s]")
    ax.grid()

    #plot uncertainties

    #output variance contribution at each time point
    var_out = Value.var(ot, [Kd, Kp, Ki])

    #standard deviation bounds
    upper_bound = Value.numeric(ot) + np.sqrt(var_out)
    lower_bound = Value.numeric(ot) - np.sqrt(var_out)

    #plot with uncertainty bounds
    fig, ax = plt.subplots(nrows=1, figsize=(8, 4), tight_layout=True, dpi=120)

    ax.plot(time, sp, lw=2, c="tab:red", label="setpoint")
    ax.plot(time, ot, lw=2, c="tab:blue", label="output")

    ax.fill_between(time, lower_bound, upper_bound, color="tab:blue", alpha=0.25, label='±1σ', ec=None)
    ax.set_xlabel('time [s]')
    ax.legend()
    ax.grid(True)


    plt.show()