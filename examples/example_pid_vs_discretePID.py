#########################################################################################
##
##                PathSim Example for PID-controller VS Discrete PID
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope, PID, Wrapper
from pathsim.solvers import RKCK54


class DiscretePID(Wrapper):
    """
    Discrete PID controller

    Parameters
    ----------
    T : float
        sampling period for the PID controller
    tau : float
        delay on Schedule event (see Schedule class)
    Kp : float
        proportional gain
    Ki : float
        integral gain
    Kd : float
        derivative  gain

    Attributes
    ----------
    Kp : float
        proportional gain
    Ki : float
        integral gain
    Kd : float
        derivative  gain
    integral : float
        integral value for the PID controller
    prev_error : float
        previous error value for derivative part in the PID controller

    """
    
    def __init__(self, T=1, tau=0, Kp=1, Ki=1, Kd=1):
        super().__init__(T=T, tau=tau)
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
    
    def func(self, error):

        """
        Run the PID controller

        Parameters
        ----------
        error : float
            error signal

        Returns
        -------
        output : float
            output of PID controller to correct the system
        """
        self.integral += error * self.T
        derivative = (error - self.prev_error) / self.T if self.T != 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        
        return output


# SYSTEM SETUP AND SIMULATION ===========================================================

#plant gain
K = 0.4

#pid parameters
Kp, Ki, Kd = 1.5, 0.5, 0.1

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
sco = Scope(labels=[r"$\epsilon(t)$", r"$\epsilon _d(t)$"])

spt2 = Source(f_s)  
err2 = Adder("+-")
dis_pid = DiscretePID(T=1,tau=0, Kp=Kp, Ki=Ki, Kd=Kd)
pnt2 = Integrator()
pgn2 = Amplifier(K)
sco2 = Scope(labels=["u(t)", "u_dis(t)"])

blocks = [spt, err, pid, pnt, pgn, sco, dis_pid, sco2, spt2, err2, pnt2, pgn2]

connections = [
    Connection(spt, err),
    Connection(pgn, err[1]),
    Connection(err, pid, sco[0]),
    Connection(pid, pnt, sco2[0]),
    Connection(pnt, pgn),
    Connection(spt2, err2),
    Connection(pgn2, err2[1]),
    Connection(err2, dis_pid, sco[1]),
    Connection(dis_pid, pnt2, sco2[1]),
    Connection(pnt2, pgn2),
]


#simulation initialization
Sim = Simulation(blocks, connections, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(100)

    sco.plot(lw=2)
    sco2.plot(lw=2)

    #plot sensitivities
    plt.show()
