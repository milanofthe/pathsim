#########################################################################################
##
##                                  SAR ADC Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import Adder, Scope, Source
from pathsim.blocks.mixed import SampleHold, Comparator, DAC


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
            comparator_result = self.inputs.get(0, 0)

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


# SYSTEM SETUP AND SIMULATION ===========================================================

n = 6                  #number of bits
f_clk = 20             #sampling frequency
T_clk = 1.0 / f_clk    #sampling period

#blocks that define the system
src = Source(lambda t: np.sin(2*np.pi*t) * np.cos(3*np.pi*t)) 
sah = SampleHold(T=T_clk) 
sub = Adder("+-")
cpt = Comparator(span=[0, 1])
dac = DAC(n_bits=n, T=T_clk/n, tau=T_clk*0.002) 
sar = SAR(n_bits=n, T=T_clk, tau=T_clk*0.001)
sco = Scope(labels=["src", "sah", "dac"]) 

blocks = [src, cpt, dac, sar, sah, sub, sco]

#connections between the blocks
connections = [
    Connection(src, sah, sco[0]),  
    Connection(sah, sub[0], sco[1]),     
    Connection(dac, sub[1], sco[2]),     
    Connection(sub, cpt),
    Connection(cpt, sar)
]

for i in range(n):
    connections.append(
        Connection(sar[i], dac[i])
        )


#simulation with adaptive solver
Sim = Simulation(
    blocks,
    connections,
    dt_max=T_clk*0.1,
    Solver=RKBS32
)


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(1)
    Sim.plot()

    plt.show()