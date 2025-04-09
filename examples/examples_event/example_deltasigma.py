#########################################################################################
##
##                                  Delta-Sigma ADC
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks.mixed import SampleHold, DAC, Comparator
from pathsim.blocks import Integrator, Adder, Scope, Source, Constant  
from pathsim.solvers import RKBS32


# SYSTEM SETUP AND SIMULATION ===========================================================

v_ref = 1.0          
f_clk = 1000          
T_clk = 1.0 / f_clk  

src = Source(lambda t: np.sin(2*np.pi*t))

sub = Adder("+-")
itg = Integrator() 
sah = SampleHold(T=T_clk, tau=T_clk*0.01)
qtz = Comparator(span=[0, 1])
dac = DAC(n_bits=1, span=[-v_ref, v_ref], T=T_clk, tau=T_clk*0.02)

sco = Scope(labels=["src", "itg", "qtz", "dac", "sah"]) 

blocks = [src, sub, itg, sah, qtz, dac, sco]

connections = [
    Connection(src, sub[0], sco[0]),  
    Connection(dac, sub[1], sco[3]),     
    Connection(sub, itg),          
    Connection(itg, sah, sco[1]),      
    Connection(sah, qtz, sco[4]),    
    Connection(qtz, dac[0], sco[2]),
]


Sim = Simulation(
    blocks,
    connections,
    dt_max=T_clk*0.1,
    Solver=RKBS32
)


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(0.5)
    sco.plot()

    plt.show()