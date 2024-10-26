########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf78.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RKDP87(ExplicitSolver):
    """
    13-stage 8-th order embedded Runge-Kutta method from Dormand and Prince
    with embedded 7-th order method for 8-th order truncation error estimate 
    that can be used to adaptively control the timestep. 

    This solver is a great choice if extremely high accuracy is required.
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):
        super().__init__(initial_value, 
                         func, 
                         jac, 
                         tolerance_lte_abs, 
                         tolerance_lte_rel)

        #counter for runge kutta stages
        self.stage = 0

        #flag adaptive timestep solver
        self.is_adaptive = True

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248/9719169821, 13/20, 1201146811/1299019798, 1.0, 1.0]

        #extended butcher table 
        self.BT = {0:[1/18],
                   1:[1/48, 1/16],
                   2:[1/32, 0, 3/32],
                   3:[5/16, 0, -75/64, 75/64],
                   4:[3/80, 0, 0, 3/16, 3/20],
                   5:[29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000],
                   6:[16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555],
                   7:[39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287],
                   8:[246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789],
                   9:[-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653],
                   10:[185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083],
                   11:[403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0],
                   12:[14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]}

        #coefficients for lower order solution evaluation
        bh = [13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535, 53011238/667516719, 2/45, 0]

        #coefficients for truncation error
        self.TR = [a-b for a, b in zip(self.BT[12], bh)]


    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep 
        based on local truncation error estimate and returns both
        """
        if len(self.Ks)<len(self.TR): 
            return True, 0.0, 0.0, 1.0

        #compute local truncation error
        tr = dt * sum(k*b for k, b in zip(self.Ks.values(), self.TR))

        #compute and clip truncation error, error ratio abs
        truncation_error_abs = float(np.max(np.clip(abs(tr), 1e-18, None)))
        error_ratio_abs = self.tolerance_lte_abs / truncation_error_abs

        #compute and clip truncation error, error ratio rel
        if np.any(self.x == 0.0): 
            truncation_error_rel = 1.0
            error_ratio_rel = 0.0
        else:
            truncation_error_rel = float(np.max(np.clip(abs(tr/self.x), 1e-18, None)))
            error_ratio_rel = self.tolerance_lte_rel / truncation_error_rel
        
        #compute error ratio and success check
        error_ratio = max(error_ratio_abs, error_ratio_rel)
        success = error_ratio >= 1.0

        #compute timestep scale
        timestep_rescale = 0.9 * (error_ratio)**(1/8)        

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """

        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)

        #compute slope and update state at stage
        self.x = dt * sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage])) + self.x_0
        
        #error and step size control
        if self.stage < 12:
            self.stage += 1
            return True, 0.0, 0.0, 1.0
        else: 
            #last stage for error control
            self.stage = 0
            return self.error_controller(dt)