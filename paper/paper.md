---
title: 'PathSim: A Decentralized Python Framework for Dynamical System Simulation'
tags:
  - Python
  - system simulation
  - dynamical systems
  - block diagram
  - ODE solver
  - automatic differentiation
  - event handling
  - hybrid systems
  - scientific computing
  - numerical integration
authors:
  - name: Milan Rother
    orcid: 0009-0006-5964-6115 
    affiliation: 1 
affiliations:
 - name: University of Technology Braunschweig
   index: 1
date: 23 april 2025
bibliography: paper.bib 
---

# Summary

PathSim is a flexible, block-based, time-domain dynamical system simulation framework implemented in Python. It enables the modeling and simulation of complex interconnected systems using an object-oriented and decentralized architecture. This architectural choice distinguishes PathSim by distributing state and computation across individual `Block` components, promoting modularity, extensibility, and flexibility. Core components include user-defined or built-in `Block` objects encapsulating specific behaviors, `Connection` objects defining explicit data flow, and a `Simulation` object managing time evolution and coordination. Dynamic blocks possess their own numerical solver instances (`engine`) for state integration. PathSim incorporates advanced features like automatic differentiation for sensitivity analysis or gradient based optimization, discrete event handling for hybrid systems, automatic linearization, hierarchical subsystems, and a comprehensive suite of ODE solvers suitable for stiff problems. It requires only core scientific Python libraries (`numpy`, `scipy`, `matplotlib`).

# Statement of Need

Simulating dynamical systems is vital across many disciplines. PathSim meets the need for a Python-native framework combining a programmatic block-diagram approach with advanced features. Traditional simulation tools often rely on centralized solvers or compiled code, which can limit flexibility and extensibility within the Python ecosystem. PathSim's decentralized architecture offers distinct advantages: enhanced modularity (blocks are self-contained units), easier extensibility (new blocks integrate naturally without core modification), and greater flexibility in model composition and analysis. PathSim specifically addresses:

* **Accessible Hybrid System Simulation:** Integrates event detection (zero-crossing, scheduled) directly into the block-diagram paradigm, simplifying the modeling of systems with both continuous and discrete dynamics.
* **Gradient-Enabled Simulation:** Provides built-in automatic differentiation, facilitating sensitivity analysis and integration with gradient-based optimization or machine learning frameworks.
* **Unified Framework for Diverse Dynamics:** Offers a wide range of solvers, including implicit methods (ESDIRK, BDF/GEAR) for stiff systems, within a single environment.
* **Extensibility in Python:** Leverages the scientific Python ecosystem with minimal dependencies. Its architecture allows straightforward creation and integration of custom blocks.

PathSim provides a powerful, flexible, and extensible open-source tool for simulating complex dynamical systems in Python.

# Architecture and Design

PathSim employs a decentralized, object-oriented design centered around three primary components:

1.  **Blocks (`Block`):** Represent individual system components or operations. They encapsulate their parameters and, if stateful (like `Integrator` or `ODE`), manage their own internal state via a dedicated numerical integration engine (`engine`) instance. This contrasts with centralized approaches where a single solver manages all system states. Blocks define `update` methods for algebraic computations within a timestep and `step`/`solve` methods for interacting with their `engine` for state evolution.
2.  **Connections (`Connection`):** Define the explicit data flow pathways between block output ports and input ports, mirroring the connections in a block diagram.
3.  **Simulation (`Simulation`):** Coordinates the overall simulation process. It maintains the list of blocks and connections. During each time step, it manages a fixed-point iteration loop. In this loop, `Connection.update()` propagates output values to inputs, and `Block.update()` computes algebraic outputs based on current inputs and states. This iterative process resolves algebraic loops and ensures consistency across interconnected blocks. The `Simulation` object then triggers the `step` (for explicit solvers) or `solve` (for implicit solvers) methods of the blocks' engines to advance their internal states. It also manages the event handling system.

This decentralized design promotes modularity, as blocks are largely self-contained. It simplifies adding new block types without altering the core simulation loop and provides flexibility in configuring individual block behaviors.

# Example Usage

The following examples demonstrate PathSims core functionalities and features.


## Continuous Nonlinear Dynamics - Pendulum

```python
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Function, Adder, Scope
from pathsim.solvers import RKBS32

#initial angle and angular velocity
phi0, omega0 = 0.9*np.pi, 0

#parameters (gravity, length)
g, l = 9.81, 1

#blocks that define the system
in1 = Integrator(omega0) 
in2 = Integrator(phi0) 
amp = Amplifier(-g/l) 
fnc = Function(np.sin) 
sco = Scope(labels=[r"$\omega$", r"$\varphi$"])

#connections between the blocks
connections = [
    Connection(in1, in2, sco[0]), 
    Connection(in2, fnc, sco[1]),
    Connection(fnc, amp), 
    Connection(amp, in1)
    ]

#simulation instance from the blocks and connections
sim = Simulation(
    blocks, 
    connections, 
    dt=0.1,  
    Solver=RKBS32, 
    tolerance_lte_rel=1e-6, 
    tolerance_lte_abs=1e-8
    )

#run the simulation for 15 seconds
sim.run(duration=15)

#read the results directly from the scope
time, [omega, phi] = sco.read()

#plot the results for quick visualization
sco.plot(".-")
sco.plot2D()
```


## Stiff Systems and Subsystems - Van der Pol

```python
from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import Integrator, Scope, Function
from pathsim.solvers import ESDIRK43

#initial conditions
x1_0, x2_0 = 2, 0

#van der Pol parameter
mu = 1000

#blocks that define the system
sco = Scope()

#subsystem with two separate integrators to emulate ODE block
ifx = Interface()
in1 = Integrator(x1_0)
in2 = Integrator(x2_0)
fnc = Function(lambda x1, x2: mu*(1 - x1**2)*x2 - x1)

sub_blocks = [ifx, in1, in2, fnc]
sub_connections = [
    Connection(in2, in1, fnc[1], ifx[1]), 
    Connection(in1, fnc, ifx), 
    Connection(fnx, in2)
    ]

#the subsystem acts just like a normal block
vdp = Subsystem(sub_blocks, sub_connections)

#blocks of the main system
blocks = [vdp, sco]

#the connections between the blocks in the main system
connections = [
    Connection(vdp, sco)
    ]

#initialize simulation with the blocks and connections
sim = Simulation(
    blocks, 
    connections, 
    Solver=ESDIRK43, 
    tolerance_lte_abs=1e-6, 
    tolerance_lte_rel=1e-3,
    tolerance_fpi=1e-9
    )

#run the simulation
sim.run(3*mu)

#quickly visualize the results
sco.plot(".-")
```


## Event Handling - Bouncing Ball


## Sensitivity Analysis - PID Controler




# References

