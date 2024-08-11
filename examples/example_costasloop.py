#########################################################################################
##
##                PathSim Example for an RF System, a Costas Loop
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np 
import matplotlib.pyplot as plt


from pathsim import Simulation, Connection

from pathsim.solvers import (
    SSPRK22,
    SSPRK33,
    BDF2,
    BDF3,
    DIRK2,
    ESDIRK43
    )

#some available blocks
from pathsim.blocks import (
    Source, 
    Amplifier, 
    Adder, 
    Multiplier, 
    Scope, 
    Spectrum, 
    Function, 
    Delay, 
    TransferFunction
    )

#some special rf blocks
from pathsim.blocks.rf import (
    ButterworthLowpassFilter, 
    ButterworthHighpassFilter, 
    ButterworthBandpassFilter,
    SquareWaveSource,
    SinusoidalSource
    )



# Parameters for simulation =========================================================================

# Simulation timestep
# dt = 1e-12
dt = 2e-12  


# Total Simulation time
#T = 85e-8
# T = 100e-9
T = 200e-9
# T = 300e-9


# Frequency of Carrier Sine
# Old version
#freq = 1e9
freq = 18.8e9


# Center frequency of the VCO
# Old value for 1e9 carrier frequency and slightly detuned VCO
#VCOTune = 0.998e9

# Slightly detuned VCO from carrier frequency (18.8GHz)
VCOTune = 1879e7
# VCOTune = 18.8e9

# Frequency of the BPSK CLK signal
#ModFreq = 5e7 ## old version
ModFreq = 250e6


# Cutoff frequency of the Low pass filter
LPF_freq = 5e9
#LPF_freq = 1e9

#HPF_freq = 5e6


# time constant for transmission delay 
# Actual delay time for later: 30cm transmission line == 100e-12s delay  
Channel_Delay = 5e-9

# time constants for loop control filter

#tau1 =  0.1
#tau1 = 5e5
#tau1 = 8e5
#tau1 = 32
#tau1 = 20e-9
tau1 = 789.56e-9


#tau2 = 0.2
#tau2 = 2e-2
#tau2 = 5e-9
tau2 = 31.41e-9

"""
Delay for phase shift to create Q arm signal; 90 degree phase shift at VCO tune frequency
PhaseShift needs to be 3 times 1/4th of a period
This is because the sine sources produce a sine; the Q arm signal needs to be a cosine
SO either use a -90degree phase shift, or delay until the cosine is reached again, as is done here
"""

PhaseShift90 = 3/(4*VCOTune)



# Block initialization ==============================================================================


# Src_Carrier = Source(lambda t : np.sin(2*np.pi*freq*t))
Src_Carrier = SinusoidalSource(freq)
# Src_BPSK = Source(lambda t : np.sign(np.sin(2*np.pi*ModFreq*t)))
Src_BPSK = SquareWaveSource(ModFreq)


# Transmission delay between signal source and control loop -----------------------------------------
TransDelay = Delay(tau=Channel_Delay) 

# Delay to simulate the 90 degree phase shift -------------------------------------------------------
PhaseShift = Delay(tau=PhaseShift90)

#Phase Detectors/ multipliers used in the arms ------------------------------------------------------
PhaseDectI = Multiplier()
PhaseDectQ = Multiplier()

# Multiplier to modulate the transmitted signal -----------------------------------------------------
Modulator = Multiplier()
LoopMult = Multiplier()


# Low pass filter for I & Q arm ---------------------------------------------------------------------

LPF_omega = 2*np.pi*LPF_freq
#LPF_omega = 3.0883e9

LPFxI = ButterworthLowpassFilter(LPF_freq, 3)
LPFxQ = ButterworthLowpassFilter(LPF_freq, 3)




# Loop control filter -------------------------------------------------------------------------------


# Loop_poles = [0.0]

# #Loop_residues = tau1
# Loop_residues = [1/tau1] 
# Loop_Const  = tau2/tau1

#this makes more sense
Loop_poles = [-1/tau1]
Loop_residues = [1/tau1] 
Loop_Const  = 0.0

LoopFilter = TransferFunction(Poles=Loop_poles, Residues=Loop_residues, Const=Loop_Const)



# VCO functions and helpers -------------------------------------------------------------------------

VCO_Fundamental = VCOTune
VCOSc = Source(lambda t: t)

## VCO Scaling should be dimensioned in a way, so that it corresponds to a real VCO
## Appropriate scaling might be: 1GHz/1V
VCO_Scaling = 1e9
#VCO_Scaling = 1
Scale = 0.8
# The Scale factor inside the VCO function might need to be adjusted according 
# to the control loop theory and might need to be lower than 1
# VCOx = Function(lambda a, b : np.cos(2*np.pi*(VCO_Fundamental + VCO_Scaling*a*Scale)*b))
VCOx = Function(lambda a, b : np.sin(2*np.pi*(VCO_Fundamental + VCO_Scaling*a*Scale)*b))
## According to this block setup, the K_0 term in theory should be equal to K_0 =  (0.8x1e9)



# Visualization blocks ------------------------------------------------------------------------------

Spc = Spectrum(labels=["BPSK", "Output I arm", "Carrier"],  # label for the visualization
               freq=np.logspace(8, 11, 1000),    # frequencies to evaluate
               t_wait=6e-8)                      # time delay until spectrum is computed


Sco = Scope(labels=["Carrier", 
                    "BPSK", 
                    "Modulated transmission signal", 
                    "Transmission Delayed Input", 
                    "Demodulated signal I arm", 
                    "Demodulated signal Q arm", 
                    "Output I arm", 
                    "Output Q arm", 
                    "Multiplied output signals Control loop", 
                    "VCO Tune voltage", 
                    "VCO Output"]
            )



# declaration of used blocks
blocks = [Src_Carrier, Src_BPSK, Modulator, TransDelay, 
          PhaseDectI, LPFxI, LoopMult, LoopFilter, VCOSc, 
          VCOx, PhaseDectQ, LPFxQ, PhaseShift, Sco, Spc]


# Connection initialization =========================================================================

connections = [
    Connection(Src_Carrier, Modulator, Sco, Spc[2]),
    Connection(Src_BPSK, Modulator[1], Sco[1], Spc),
    Connection(Modulator, TransDelay, Sco[2]),
    Connection(TransDelay, PhaseDectI, PhaseDectQ, Sco[3]),
    Connection(PhaseDectQ, LPFxQ, Sco[5]),
    Connection(PhaseDectI, LPFxI, Sco[4]),
    Connection(LPFxI, LoopMult, Sco[6], Spc[1]),
    Connection(LPFxQ, LoopMult[1], Sco[7]),
    Connection(LoopMult, LoopFilter, Sco[8]),
    Connection(LoopFilter, VCOx, Sco[9]),
    Connection(VCOSc, VCOx[1]),
    Connection(VCOx, PhaseShift, PhaseDectI[1], Sco[10]),
    Connection(PhaseShift, PhaseDectQ[1])
    ]



# Simulation initialization =========================================================================

# Simulation setup
Sim = Simulation(blocks, connections, dt, log=True, Solver=SSPRK22)

# Main simulation run command
Sim.run(T)

#plot scope recordings
Sco.plot()

#plot spectrum
Spc.plot()

plt.show()
