# PathSim Development Roadmap

This is a preliminary roadmap for PathSim's development. To be fair, its more a collection of ideas and potential directions then a real roadmap.


## Solvers

- multistep methods (BDF, GEAR) need startup methods to build history


## Analysis

- periodic steady state solver, probably shooting method with automatic frequency/period detection
- small signal frequency domain analysis based on linearized system
- time constant extraction via eigenvalues of linearized dynamic blocks


## Performance

- fork-join block-level parallelization with Python 3.13 free-threading
- more robust and adaptable steady state solver, perhaps through damping, or using `EUB` with convergence driven timestep control
- exponential integrators for LTI blocks (`StateSpace`, `TransferFunction`) and for linearized dynamic blocks
- more robust resolution of algebraic loops (currently implicitly handled through fixed point loop in `Simulation._update()`), probably: loop detection, accelerator injection
- jit compilation of internal functions in the operators (`Operator`, `DynamicOperator`)


## API

- separate transient and steady state analysis results for scope and spectrum
- add options to integrators to specify the type of optimizer to be used to solve the implicit update equation, currenly the hybrid `NewtonAnderson` is used, but more flexibility might be nice in the future


## Testing

- complete testing for all the blocks (currently testing for blocks is mostly top down) including linearization
- test automatic differentiation through `Value` with every block and with linearization


## Integration

- support for Modelica / FMI models, import as special blocks
- special circuit blocks, generated from SPICE netlists