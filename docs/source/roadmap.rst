.. _ref-roadmap:

Development Roadmap
===================
This is a preliminary roadmap for PathSim's development. To be fair, its more a collection of ideas and potential directions then a real roadmap. The items are ordered by category. If you want to contribute (check out the :ref:`ref-contributing` guidelines), you can take on one or more of them!

Blocks
------
- adding more convenience blocks, for example math operations or different kinds of sources 

Solvers
-------
- make solver term :math:`\dot{x}` accessible for solving index 1 DAEs, especially for stiffly accurate :class:`.DIRK` and :class:`.ESDIRK` methods, will be the basis for future `DAE` blocks
- add interpolant within timestep to solvers for dense output and to improve interpolation for the event mechanism 

Analysis
--------
- periodic steady state solver, probably shooting method with automatic frequency/period detection
- small signal frequency domain analysis based on linearized system

Performance
-----------
- fork-join block-level parallelization with Python 3.13 free-threading
- more robust and adaptable steady state solver, perhaps through damping, or using `EUB` with convergence driven timestep control
- exponential integrators for LTI blocks (:class:`.StateSpace`, :class:`.TransferFunction`) and for linearized dynamic blocks
- jit compilation of internal functions in the operators (:class:`.Operator`, :class:`.DynamicOperator`)

Documentation
-------------
- testing of code blocks in the :ref:`ref-examples` section through either `jupyter-book` integration, or `doctest`
- add the standalone examples from the `repo <https://github.com/milanofthe/pathsim/blob/master/examples>`_ to the :ref:`ref-examples` section in the docs with additional explanations as for the other examples
- the integrators (:class:`.Solver` child classes) would benefit from more descriptive docstrings for the api reference including explanations and references (literature) to the specific method used and also some usage suggestions to make solver choice easier and more transparent for the user
- add more tutorials to the docs, that focus on:
   - types of analyses and visualization methods (transient -> :class:`.Scope`, frequency -> :class:`.Spectrum`, steadystate -> ??)
   - types of available solvers in the PathSim solver suite, *find your own solver*
   - the block diagram modelling paradigm in general
   - hierarchical modeling with the :class:`.Subsystem` class
- type hints for everything

API
---
- separate the different kinds of analyses more clearly, transient and steady state analysis results should be separated for :class:`.Scope` and :class:`.Spectrum` blocks to feel more natural
- add options to integrators (:class:`.ImplicitSolver`) to specify the type of optimizer to be used to solve the implicit update equation, currenly the hybrid `NewtonAnderson` is used, but more flexibility might be nice in the future

User Interface & Visualization
------------------------------
- visualization of the connection graph from PathSim models for debugging as an intermediate solution, before a fully fledged GUI is available, maybe using `graphviz`
- improved and faster interactive plotting, moving to from matplotlib to plotly? 
- block diagram editor user interface as an extension to PathSim, perhaps using React Flow

Cross Compatibility
-------------------
- support for FMI / FMU (model exchange), import and export of PathSim models and blocks 
- support for electrical circuits, SPICE netlists 
- support for s-parameters (touchstone files) by vectorfitting and wrapping :class:`.StateSpace` block

Testing
-------
- complete testing for blocks in :mod:`.pathsim/blocks/rf` and :mod:`.pathsim/blocks/mixed` (currently testing for these blocks is mostly top down)
- test automatic differentiation through `Value` with every block and with linearization
- permutation testing for all kinds of system topologies, blocks and solvers
