# PathSim Development Roadmap

This is a preliminary roadmap for PathSim's development. To be fair, its more a collection of ideas and potential directions then a real roadmap. 


# Algorithms

## Solvers

- multistep methods (`BDF`, `GEAR`) need startup methods to build history, to maintain consistency order globally
- make solver term $\\dot{x}$ accessible for solving index 1 DAEs, especially for stiffly accurate `DIRK` and `ESDIRK` methods, will be the basis for future `DAE` blocks
- add interpolant within timestep to solvers for dense output and to improve interpolation for the event mechanism 

## Events

- separate checks for event types in simulation loop, purely time dependent events (`Schedule`) can be estimated before the timestep is taken and therefore be approached without backtracking, which would improve performance (maybe add some method `Event.estimate(t)`)

## Analysis

- periodic steady state solver, probably shooting method with automatic frequency/period detection
- small signal frequency domain analysis based on linearized system

## Performance

- fork-join block-level parallelization with Python 3.13 free-threading
- more robust and adaptable steady state solver, perhaps through damping, or using `EUB` with convergence driven timestep control
- exponential integrators for LTI blocks (`StateSpace`, `TransferFunction`) and for linearized dynamic blocks
- more robust resolution of algebraic loops (currently implicitly handled through fixed point loop in `Simulation._update()`), probably: loop detection, accelerator injection
- jit compilation of internal functions in the operators (`Operator`, `DynamicOperator`)


# Usability

## Documentation

- the integrators (`Solver` child classes) would benefit from more descriptive docstrings for the api reference including explanations and references (literature) to the specific method used and also some usage suggestions to make solver choice easier and more transparent for the user
- add more tutorials to the docs, that focus on:
	- types of analyses and visualization methods (transient -> `Scope`, frequency -> `Spectrum`, steadystate -> ??)
	- types of available solvers in the PathSim solver suite, *find your own solver*
	- the block diagram modelling paradigm in general
	- hierarchical modeling with the `Subsystem` class
- type hints for everything

## API

- separate the different kinds of analyses more clearly, transient and steady state analysis results should be separated for `Scope` and `Spectrum` blocks to feel more natural
- add options to integrators (`ImplicitSolver`) to specify the type of optimizer to be used to solve the implicit update equation, currenly the hybrid `NewtonAnderson` is used, but more flexibility might be nice in the future

## Cross Compatibility

- support for FMI / FMU, import and export of PathSim models and blocks 
- support for electrical circuits, SPICE netlists 
- support for s-parameters (touchstone files) by vectorfitting and wrapping `StateSpace` block


# Testing

## Blocks
- complete testing for blocks in `pathsim/blocks/rf` and `pathsim/blocks/mixed` (currently testing for these blocks is mostly top down)

## Features
- test automatic differentiation through `Value` with every block and with linearization

## Robustness
- permutation testing for all kinds of system topologies, blocks and solvers
