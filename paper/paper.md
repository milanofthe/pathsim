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
 - name: Technische Universität Braunschweig
   index: 1
date: 23 april 2025
bibliography: paper.bib 
---

# Summary

PathSim is a flexible, block-based, time-domain dynamical system simulation framework implemented in Python. It enables the modeling and simulation of complex interconnected systems using an object-oriented and decentralized architecture. This architectural choice distinguishes PathSim by distributing state and computation across individual `Block` components, promoting modularity, extensibility, and flexibility. Core components include user-defined or built-in `Block` objects encapsulating specific behaviors, `Connection` objects defining explicit data flow, and a `Simulation` object managing time evolution and coordination. Dynamic blocks possess their own numerical solver instances (`engine`) for state integration. PathSim incorporates advanced features like automatic differentiation for sensitivity analysis or gradient based optimization, discrete event handling for hybrid systems, automatic linearization, hierarchical subsystems, and a comprehensive suite of ODE solvers suitable for stiff problems. It requires only core scientific Python libraries: NumPy [@harris2020array], SciPy [@virtanen2020scipy], and Matplotlib [@hunter2007matplotlib].

# Statement of Need

Simulating dynamical systems is vital across many disciplines. PathSim meets the need for a Python-native framework combining a programmatic block-diagram approach with advanced features. Traditional simulation tools often rely on centralized solvers or compiled code, which can limit flexibility and extensibility within the Python ecosystem. PathSim's decentralized architecture offers distinct advantages: enhanced modularity (blocks are self-contained units), easier extensibility (new blocks integrate naturally without core modification), and greater flexibility in model composition and analysis, Cosimulation and Hardware in the Loop (HiL) testing. PathSim specifically addresses:

* **Accessible Hybrid System Simulation:** Integrates event detection (zero-crossing, scheduled) directly into the block-diagram paradigm, simplifying the modeling of systems with both *continuous and discrete dynamics*.
* **Gradient-Enabled Simulation:** Provides built-in *automatic differentiation* for sensitivity analysis and integration with gradient-based optimization or machine learning frameworks.
* **Unified Framework for Diverse Dynamics:** Offers a wide range of solvers, including implicit methods (ESDIRK, BDF/GEAR) for *stiff systems*.
* **Extensibility in Python:** Leverages the scientific Python ecosystem with minimal dependencies. Its architecture allows straightforward creation and integration of custom blocks.

PathSim provides a powerful, flexible, and extensible open-source tool for simulating complex dynamical systems with minimal dependencies in Python.

# Comparison to Existing Tools

Several Python tools exist for simulating dynamical systems. Standard ODE solvers like `scipy.integrate.solve_ivp` [@virtanen2020scipy] offer robust integration but lack a structured framework for modeling complex, interconnected systems or handling discrete events natively. Libraries like `SimuPy` [@Margolis2017; @SimuPyRepo] provide a block-based modeling approach similar to PathSim, leveraging SymPy for symbolic definition and SciPy solvers for integration. Other frameworks like `Collimator` [@pycollimator] offer graphical interfaces and JAX-based acceleration but introduce dependencies beyond the standard scientific Python stack. `bdsim` [@bdsimRepo] also provides block diagram simulation with a graphical editor, focusing often on robotics. PathSim differentiates itself by offering a purely script-based block-diagram interface with a *decentralized architecture*, *native integration* of both automatic differentiation and discrete event handling, and a *built-in library* of independently implemented and verified ODE solvers (beyond wrapping SciPy), all while maintaining *minimal core dependencies*.

# Architecture and Design

PathSim employs a decentralized, object-oriented design centered around three primary components:

1.  **Blocks (`Block`):** Represent individual system components or operations. They encapsulate their parameters and, if stateful (like `Integrator` or `ODE`), manage their own internal state via a dedicated numerical integration engine (`engine`) instance. This contrasts with centralized approaches where a single solver manages all system states. Blocks define `update` methods for algebraic computations within a timestep and `step`/`solve` methods for interacting with their `engine` for state evolution.
2.  **Connections (`Connection`):** Define the explicit data flow pathways between block output ports and input ports, mirroring the connections in a block diagram.
3.  **Simulation (`Simulation`):** Coordinates the overall simulation process. It maintains the list of blocks and connections. During each time step, it manages a fixed-point iteration loop. In this loop, `Connection.update()` propagates output values to inputs, and `Block.update()` computes algebraic outputs based on current inputs and states. This iterative process resolves algebraic loops and ensures consistency across interconnected blocks. The `Simulation` object then triggers the `step` (for explicit solvers) or `solve` (for implicit solvers) methods of the blocks' engines to advance their internal states. It also manages the event handling system.

The decentralized design promotes modularity, as blocks are fully self-contained. It simplifies adding new block types without altering the core simulation loop and provides flexibility in configuring individual block behaviors. Additionaly this opens up integration with other simulation environments (co-simulation), or hardware in the loop (HiL) setups through encapsulation within blocks.

# Example Usage

The following example of a nonlinear pendulum demonstrate PathSims core system modeling and simulation flow.

![Mechanical model and block diagram of nonlinear pendulum.](assets/pendulum_block_diagram.svg)

```python
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Function, Adder, Scope
from pathsim.solvers import RKCK54

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

#simulation instance from the blocks and connections
sim = Simulation(
    blocks=[in1, in2, amp, fnc, sco], 
    connections=[
        Connection(in1, in2, sco[0]), 
        Connection(in2, fnc, sco[1]),
        Connection(fnc, amp), 
        Connection(amp, in1)
        ], 
    Solver=RKCK54,
    tolerance_lte_rel=1e-6,
    tolerance_lte_abs=1e-8
    )

#run the simulation for 15 seconds
sim.run(duration=15)

#read the results directly from the scope for postprocessing
time, [omega, phi] = sco.read()

#plot the results for quick visualization
sco.plot(".-")
sco.plot2D()
```

![Time series results from `Scope.plot()` ](assets/pendulum_result_timeseries.svg) ![2D phase portrait from `Scope.plot2D()`](assets/pendulum_result_phasespace.svg)

This code shows block instantiation, connection definition, simulation setup (including solver selection), execution, and result visualization. Full examples demonstrating event handling, stiff systems, and sensitivity analysis are available in the software repository [@PathSimRepo] and documentation [@PathSimDocs].


# References

