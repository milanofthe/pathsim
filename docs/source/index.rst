.. pathsim documentation master file, created by
   sphinx-quickstart on Wed Feb 19 13:55:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PathSim: Differentiable System Simulation 
=========================================

**PathSim** is a flexible block-based time-domain system simulation framework in Python with automatic differentiation capabilities and an event handling mechanism. It provides a variety of classes that enable modeling and simulating complex interconnected dynamical systems through Python scripting.

Key Features:

- **Hot-swappable** blocks and solvers during simulation
- Blocks are inherently **MIMO** (Multiple Input, Multiple Output) capable
- Wide range of **numerical integrators** (implicit, explicit, high order, adaptive)
- **Modular and hierarchical** modeling with (nested) subsystems
- **Event handling** system to detect and resolve discrete events (zero-crossing detection)
- Automatic differentiation for **fully differentiable** system simulations
- **Extensibility** by subclassing the base `Block` class and implementing just a handful of methods

The source code can be found in the `GitHub repository <https://github.com/milanofthe/pathsim>`.


Quickstart
----------

1. Install PathSim with pip:

.. code-block:: bash

    pip install pathsim


2. Build and simulate a system unsing blocks and connections:

.. code-block:: python
    
    import numpy as np

    from pathsim import Simulation, Connection
    from pathsim.blocks import Source, Integrator, Scope

    Sr = Source(np.sin)
    In = Integrator()
    Sc = Scope()

    Sim = Simulation(
        blocks=[Sr, In, Sc], 
        connections=[
            Connection(Sr, In),
            Connection(Sr, Sc[0]),
            Connection(In, Sc[1]),
            ], 
        dt=0.01 
        )

    Sim.run(10)

    Sc.plot()


Table of Contents
-----------------

This documentation is structured in the following way:

.. toctree::
   :maxdepth: 2

   api
   structure
   examples


An Example
----------

Here's a simple example of a linear feedback system, simulated with PathSim. 

.. image:: figures/linear_feedback_blockdiagram.png
   :width: 700
   :align: center
   :alt: block diagram of linear feedback system


The block diagramm can be translated to a netlist by using the blocks and the connection class provided by PathSim:

.. code-block:: python

    from pathsim import Simulation, Connection
    from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

    #values parameters
    a, b, x0 = -1, 1, 2

    #step function
    tau = 3
    def s(t):
        return b*int(t>tau)

    #blocks that define the system
    Src = Source(s)
    Int = Integrator(x0)
    Amp = Amplifier(a)
    Add = Adder()
    Sco = Scope(labels=["step", "response"])

    blocks = [Src, Int, Amp, Add, Sco]

    #the connections between the blocks
    connections = [
        Connection(Src, Add[0], Sco[0]),
        Connection(Amp, Add[1]),
        Connection(Add, Int),
        Connection(Int, Amp, Sco[1])
        ]

    #initialize simulation with the blocks, connections, timestep
    Sim = Simulation(blocks, connections, dt=0.01, log=True)
        
    #run the simulation for some time
    Sim.run(4*tau)

    #plot the results from the scope
    Sco.plot()


.. image:: figures/linear_feedback_result.png
   :width: 700
   :align: center
   :alt: simulation result of linear feedback system


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`