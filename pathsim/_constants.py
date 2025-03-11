##################################################################################
##
##                  GLOBAL CONSTANTS AND TOLERANCES FOR PATHSIM
##                               (_constants.py)
##
##                              Milan Rother 2025
##
##################################################################################

# Global floating point tolerance
TOLERANCE_FLOAT = 1e-16

# Simulation default constants
SIM_DT = 0.01                     # simulator timestep (initial) 
SIM_DT_MIN = 1e-16                # minimum simulator timestep
SIM_DT_MAX =  None                # maximum simulator timestep
SIM_TOLERANCE_FPI = 1e-12         # tolerance for fixed point loop convergence
SIM_ITERATIONS_MIN = 1            # minimum number of fixed point loop iterations
SIM_ITERATIONS_MAX = 200          # maximum number of fixed point loop iterations

# Solver default constants
SOLVER_TOLERANCE_LTE_ABS = 1e-8   # absolute local truncation error (adaptive solvers)
SOLVER_TOLERANCE_LTE_REL = 1e-5   # relative local truncation error (adaptive solvers)
SOLVER_TOLERANCE_FPI = 1e-12      # tolerance for optimizer convergence (implicit solvers)
SOLVER_ITERATIONS_MAX = 500       # maximum number of optimizer iterations (implicit solvers)
SOLVER_SCALE_MIN = 0.1            # maximum timestep rescale factor (adaptive solvers)
SOLVER_SCALE_MAX = 10             # minimum timestep rescale factor (adaptive solvers)
SOLVER_BETA = 0.9                 # savety for timestep control (adaptive solvers)

# Event default constants
EVENT_TOLERANCE = 1e-4            # tolerance for event detection