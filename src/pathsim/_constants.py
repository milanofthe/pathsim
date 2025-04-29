##################################################################################
##
##                  GLOBAL CONSTANTS AND TOLERANCES FOR PATHSIM
##                               (_constants.py)
##
##                              Milan Rother 2025
##
##################################################################################

# global floating point tolerance ------------------------------------------------

TOLERANCE = 1e-16

# simulation default constants ---------------------------------------------------

SIM_TIMESTEP = 0.01         # simulator timestep (initial) 
SIM_TIMESTEP_MIN = 1e-16    # minimum simulator timestep
SIM_TIMESTEP_MAX =  None    # maximum simulator timestep
SIM_TOLERANCE_FPI = 1e-12   # tolerance for fixed point loop convergence
SIM_ITERATIONS_MIN = None   # minimum number of fixed point loop iterations -> determined automatically
SIM_ITERATIONS_MAX = 200    # maximum number of fixed point loop iterations


# solver default constants -------------------------------------------------------

SOL_TOLERANCE_LTE_ABS = 1e-8   # absolute local truncation error (adaptive solvers)
SOL_TOLERANCE_LTE_REL = 1e-5   # relative local truncation error (adaptive solvers)
SOL_TOLERANCE_FPI = 1e-9       # tolerance for optimizer convergence (implicit solvers)
SOL_ITERATIONS_MAX = 200       # maximum number of optimizer iterations (implicit solvers)
SOL_SCALE_MIN = 0.1            # minimum timestep rescale factor (adaptive solvers)
SOL_SCALE_MAX = 10             # maximum timestep rescale factor (adaptive solvers)
SOL_BETA = 0.9                 # savety for timestep control (adaptive solvers)


# event default constants --------------------------------------------------------

EVT_TOLERANCE = 1e-4   # tolerance for event detection


# logging default constants ------------------------------------------------------

LOG_ENABLE = True        # logging is enabled by default  
LOG_MIN_INTERVAL = 1.0   # logging interval in seconds for progress, etc.
LOG_UPDATE_EVERY = 0.2   # logging update milestone every 0.2 -> every 20%


# colors for visualization -------------------------------------------------------

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
