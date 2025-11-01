# PathSim Testing Philosophy

The tests for **PathSim** use `unittest` and follow a two tier approach:

1. Tests in [tests/pathsim]() are bottom-up atomic tests for all the individual classes, their methods and functions to ensure their core functionalities do what they are supposed to. Complexity here is rather basic, tests should be as self contained as possible.

2. Tests in [tests/evals]() are top-down and implement specific dynamical systems, simulate them and compare the results to reference solutions. They test complexity, permutations, and ultimately ensure accuracy and functionality of the whole framework.
