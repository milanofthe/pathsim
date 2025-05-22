#########################################################################################
##
##                                  SAR ADC Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Adder, Scope, Source, ButterworthLowpassFilter, 
    SampleHold, Comparator, DAC
    )

from pathsim.solvers import RKBS32


# CUSTOM SAR LOGIC BLOCK ================================================================

from pathsim.blocks._block import Block
from pathsim.events import Schedule

class SAR(Block):

    def __init__(self, n_bits=4, T=1, tau=0):
        super().__init__()

        self.n_bits = n_bits
        self.T = T
        self.tau = tau

        self.register = 0
        self.trial_weight = 1 << (self.n_bits - 1)

        self.outputs = {i: 0 for i in range(self.n_bits)}

        def _step(t):
            comparator_result = self.inputs[0]

            previous_weight = (self.trial_weight << 1) if self.trial_weight > 0 else 1

            if previous_weight <= (1 << (self.n_bits -1)) and comparator_result == 0:
                self.register &= ~previous_weight 

            self.register |= self.trial_weight

            for i in range(self.n_bits):
                self.outputs[i] = (self.register >> i) & 1

            if self.trial_weight == 1: 
                self.trial_weight = 1 << (self.n_bits - 1)
                self.register = 0
            else:
                self.trial_weight >>= 1

        self.events = [
            Schedule(
                t_start=self.tau,
                t_period=self.T/self.n_bits,
                func_act=_step
                )
            ]

    def __len__(self):
        return 0


# SYSTEM SETUP AND SIMULATION ===========================================================

n = 8                 #number of bits
f_clk = 50            #sampling frequency
T_clk = 1.0 / f_clk   #sampling period

#blocks that define the system
src = Source(lambda t: np.sin(2*np.pi*t) * np.cos(5*np.pi*t)) 
sah = SampleHold(T=T_clk) 
sub = Adder("+-")
cpt = Comparator(span=[0, 1])
dac1 = DAC(n_bits=n, T=T_clk/n, tau=T_clk*2e-3) 
dac2 = DAC(n_bits=n, T=T_clk, tau=T_clk) 
lpf = ButterworthLowpassFilter(f_clk/5, n=3)
sar = SAR(n_bits=n, T=T_clk, tau=T_clk*1e-3)
sco = Scope(labels=["src", "sah", "dac1", "dac2", "lpf"]) 

blocks = [src, cpt, dac1, dac2, lpf, sar, sah, sub, sco]

#connections between the blocks
connections = [
    Connection(src, sah, sco[0]),  
    Connection(sah, sub[0], sco[1]),     
    Connection(dac1, sub[1], sco[2]),    
    Connection(dac2, lpf, sco[3]),     
    Connection(lpf, sco[4]),  
    Connection(sub, cpt),
    Connection(cpt, sar)
]

for i in range(n):
    connections.append(
        Connection(sar[i], dac1[i], dac2[i])
        )


#simulation with adaptive solver
Sim = Simulation(
    blocks,
    connections,
    # dt_max=T_clk*0.1,
    Solver=RKBS32
)


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(1)
    Sim.plot()

    plt.show()