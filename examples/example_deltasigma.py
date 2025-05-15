#########################################################################################
##
##                                  Delta-Sigma ADC
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import firwin

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Integrator, Adder, Scope, Source, 
    SampleHold, DAC, Comparator, FIR
    )

from pathsim.solvers import RKBS32


# SYSTEM SETUP AND SIMULATION ===========================================================

v_ref = 1.0           #dac reference
f_clk = 100           #sampling frequency
T_clk = 1.0 / f_clk   #sampling period

fir_coeffs = firwin(20, f_clk/50, fs=f_clk)


#blocks that define the system
src = Source(lambda t: np.sin(2*np.pi*t))
sub = Adder("+-")
itg = Integrator() 
sah = SampleHold(T=T_clk, tau=T_clk*1e-3)
qtz = Comparator(span=[0, 1])
dac = DAC(n_bits=1, span=[-v_ref, v_ref], T=T_clk, tau=T_clk*2e-3)
lpf = FIR(coeffs=fir_coeffs, T=T_clk, tau=T_clk*2e-3)
sc1 = Scope(labels=["src", "qtz", "dac", "lpf"]) 
sc2 = Scope(labels=["itg", "sah"]) 

blocks = [src, sub, itg, sah, qtz, dac, lpf, sc1, sc2]

#connections between the blocks
connections = [
    Connection(src, sub[0], sc1[0]),  
    Connection(dac, sub[1], lpf, sc1[2]),     
    Connection(sub, itg),          
    Connection(itg, sah, sc2[0]),      
    Connection(sah, qtz, sc2[1]),    
    Connection(qtz, dac[0], sc1[1]),
    Connection(lpf, sc1[3]),
]


#simulation with adaptive solver
Sim = Simulation(
    blocks,
    connections,
    dt_max=T_clk*0.1,
    Solver=RKBS32
)



# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(2)
    Sim.plot()

    plt.show()