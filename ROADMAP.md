# PathSim Development Roadmap

This is a preliminary roadmap for PathSim's development. To be fair, its more a collection of ideas or a To-Do list then a real roadmap.


## Analysis

- periodic steady state solver, probably shooting method with automatic frequency/period detection


## Performance

- fork-join block-level parallelization with Python 3.13 free-threading
- more robust and adaptable steady state solver, perhaps through damping
- exponential integrators for LTI blocks (`StateSpace`, `TransferFunction`) and for linearized dynamic blocks
- more robust resolution of algebraic loops (currently implicitly handled through fixed point loop in `Simulation.update()`), probably: loop detection, accelerator injection
- jit compilation of internal functions in the operators (`Operator`, `DynamicOperator`)


## API

- separate transient and steady state analysis resuts for scope and spectrum


## Testing

- complete testing for all the blocks (currently testing is mostly second order) including linearization

