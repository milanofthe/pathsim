##################################################################################
##
##                  GLOBAL CONSTANTS AND TOLERANCES FOR PATHSIM
##                               (_constants.py)
##
##                              Milan Rother 2025
##
##################################################################################

# Global floating point tolerance
TOLERANCE = 1e-16

# Simulation default constants
SIM_TIMESTEP = 0.01            # simulator timestep (initial) 
SIM_TIMESTEP_MIN = 1e-16       # minimum simulator timestep
SIM_TIMESTEP_MAX =  None       # maximum simulator timestep
SIM_TOLERANCE_FPI = 1e-12      # tolerance for fixed point loop convergence
SIM_ITERATIONS_MIN = 1         # minimum number of fixed point loop iterations
SIM_ITERATIONS_MAX = 200       # maximum number of fixed point loop iterations

# Solver default constants
SOL_TOLERANCE_LTE_ABS = 1e-8   # absolute local truncation error (adaptive solvers)
SOL_TOLERANCE_LTE_REL = 1e-5   # relative local truncation error (adaptive solvers)
SOL_TOLERANCE_FPI = 1e-9       # tolerance for optimizer convergence (implicit solvers)
SOL_ITERATIONS_MAX = 200       # maximum number of optimizer iterations (implicit solvers)
SOL_SCALE_MIN = 0.1            # maximum timestep rescale factor (adaptive solvers)
SOL_SCALE_MAX = 10             # minimum timestep rescale factor (adaptive solvers)
SOL_BETA = 0.9                 # savety for timestep control (adaptive solvers)

# Event default constants
EVT_TOLERANCE = 1e-4         # tolerance for event detection

# Colors for visualization
COLOR_RED = "#e41a1c"
COLOR_BLUE = "#377eb8"
COLOR_GREEN = "#4daf4a"
COLOR_PURPLE = "#984ea3"
COLOR_ORANGE = "#ff7f00"
COLORS_ALL = [
	COLOR_RED, 
	COLOR_BLUE, 
	COLOR_GREEN, 
	COLOR_PURPLE, 
	COLOR_ORANGE
	]