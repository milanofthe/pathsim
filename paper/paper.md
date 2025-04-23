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

# Example Usage: Harmonic Oscillator

The following example demonstrates simulating a damped harmonic oscillator ($\ddot{x} + \frac{c}{m} \dot{x} + \frac{k}{m} x = 0$) using PathSim's block-based approach.

```python


```

This example illustrates how the system's differential equation is translated into interconnected blocks, forming feedback loops. The Simulation object manages the execution and data recording via the Scope.


# References

