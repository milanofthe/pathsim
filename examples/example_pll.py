#########################################################################################
##
##                    PathSim Example for Phase-Locked Loop (PLL)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    SinusoidalSource, Multiplier, ButterworthLowpassFilter,
    Amplifier, Integrator, Sin, Adder, Constant, Scope, Spectrum
)
from pathsim.solvers import RKCK54


# SYSTEM SETUP AND SIMULATION ===========================================================

# Reference signal parameters
f_ref = 10.0          # reference frequency in Hz
A_ref = 1.0           # reference amplitude

# VCO parameters
f_vco_center = 9.5    # VCO center frequency in Hz (offset from reference)
K_vco = 2.0           # VCO gain in Hz/V (reduced for stability)

# Phase detector gain
K_pd = 0.3            # phase detector gain (reduced for Type-II stability)

# Loop filter parameters (proper lowpass filter)
f_cutoff = 2.0        # loop filter cutoff frequency in Hz
filter_order = 2      # loop filter order


# Blocks that define the PLL system ====================================================

# Reference signal (input to lock to)
ref = SinusoidalSource(frequency=f_ref, amplitude=A_ref)

# Phase detector (multiplier acts as mixer)
phase_detector = Multiplier()

# Phase detector gain
pd_gain = Amplifier(K_pd)

# Loop filter (Butterworth lowpass filter)
loop_filter = ButterworthLowpassFilter(Fc=f_cutoff, n=filter_order)

# Loop integrator (makes this a Type-II PLL for frequency acquisition)
loop_integrator = Integrator(initial_value=0.0)

# VCO free-running frequency contribution
vco_free = Adder()

# VCO gain (converts control voltage to frequency in rad/s)
vco_gain = Amplifier(K_vco * 2 * np.pi)

# VCO integrator (integrates frequency to get phase)
vco_integrator = Integrator(initial_value=0.0)

# VCO output (converts phase to sinusoid)
vco_output = Sin()

# Constant for VCO center frequency (in rad/s)
center_freq = Constant(f_vco_center * 2 * np.pi)

# Scope for time-domain observation
scope = Scope(labels=["Reference", "VCO Output", "Control Voltage", "Phase Error"])

# Additional scope for VCO internals
scope_vco = Scope(labels=["VCO Phase", "VCO Freq Input", "Free-run Freq"])

# Spectrum analyzer for frequency-domain observation
freq_span = np.linspace(0, 40, 1000)
spectrum = Spectrum(freq=freq_span, labels=["Reference", "VCO Output"])


blocks = [
    ref, phase_detector, pd_gain, loop_filter, loop_integrator, vco_free, vco_gain,
    vco_integrator, vco_output, center_freq, scope, scope_vco, spectrum
]


# Connections between blocks ============================================================

connections = [
    # Reference signal paths
    Connection(ref, phase_detector[0], scope[0], spectrum[0]),

    # Phase detector to gain to loop filter
    Connection(phase_detector, pd_gain, scope[3]),
    Connection(pd_gain, loop_filter),

    # Loop filter to integrator (Type-II)
    Connection(loop_filter, loop_integrator),

    # Integrator output (control voltage) to VCO
    Connection(loop_integrator, vco_gain, scope[2]),

    # VCO: gain → adder → integrator → sin function
    Connection(vco_gain, vco_free[0], scope_vco[1]),
    Connection(center_freq, vco_free[1], scope_vco[2]),
    Connection(vco_free, vco_integrator),
    Connection(vco_integrator, vco_output, scope_vco[0]),

    # VCO output feedback to phase detector and monitoring
    Connection(vco_output, phase_detector[1], scope[1], spectrum[1])
]


# Simulation initialization =============================================================

Sim = Simulation(
    blocks,
    connections,
    dt=0.01,
)


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run simulation for enough time to see lock behavior
    Sim.run(100.0)

    # Plot time-domain signals
    scope.plot(lw=1.5)
    plt.suptitle("PLL Time-Domain Signals")

    # Plot VCO internals
    scope_vco.plot(lw=1.5)
    plt.suptitle("VCO Internal Signals")

    # Plot frequency spectrum
    spectrum.plot()
    plt.suptitle("PLL Frequency Spectrum")
    plt.xlim(0, 20)

    plt.show()
