# PathSim Test Suite

The tests for PathSim use `unittest` and follow a two tier approach. 

1. Tests in [tests/pathsim]() are bottom-up atomic tests that test all the individual classes, their methods and functions for their core functionalities. Complexity here is rather basic. 

2. Tests in [tests/evals]() are top-down and implement specific dynamical systems similar to what a user would do. Here, everything comes together. They test complexity, permutations, and ultimately ensure accuracy and functionality of the whole system simulation framework.


