########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk85.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ImplicitSolver
from ..utils.funcs import numerical_jacobian


# SOLVERS ==============================================================================

class ESDIRK85(ImplicitSolver):
    """
    16 stage 8-th order L-stable, stiffly accurate, stage order 2 ESDIRK method with 
    embedded 5-th order method for stepsize control. This very high order integrator 
    is suited for very stiff problems that require very high accuracy but is also 
    relatively expensive due to the insane 15 implicit (1 explicit) stages.

    This method is a real beast and it remains to be seen how practical it is.

    FROM : 
        VERY HIGH-ORDER A-STABLE STIFFLY ACCURATE DIAGONALLY 
        IMPLICIT RUNGE-KUTTA METHODS WITH ERROR ESTIMATORS
        YOUSEF ALAMRI AND DAVID I. KETCHESON
        ESDIRK(16,8)[2]SAL-[(16,5)]
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
        self.eval_stages = [0.0,
                            0.234637638717043,
                            0.558545926594724,
                            0.562667638694992,
                            0.697898381329126,
                            0.956146958839776,
                            0.812903043340468,
                            0.148256733818785,
                            0.944650387704291,
                            0.428471803715736,
                            0.984131639774509,
                            0.320412672954752,
                            0.974077670791771,
                            0.852850433853921,
                            0.823320301074444,
                            1.0]

        #butcher table
        self.BT = {0:[0.0],
                   1:[0.117318819358521,0.117318819358521],
                   2:[0.0557014605974616,0.385525646638742,0.117318819358521],
                   3:[0.063493276428895,0.373556126263681,0.0082994166438953,0.117318819358521],
                   4:[0.0961351856230088,0.335558324517178,0.207077765910132,-0.0581917140797146,0.117318819358521],
                   5:[0.0497669214238319,0.384288616546039,0.0821728117583936,0.120337007107103,0.202262782645888,0.117318819358521],
                   6:[0.00626710666809847,0.496491452640725,-0.111303249827358,0.170478821683603,0.166517073971103,-0.0328669811542241,0.117318819358521],
                   7:[0.0463439767281591,0.00306724391019652,-0.00816305222386205,-0.0353302599538294,0.0139313601702569,-0.00992014507967429,0.0210087909090165,0.117318819358521],
                   8:[0.111574049232048,0.467639166482209,0.237773114804619,0.0798895699267508,0.109580615914593,0.0307353103825936,-0.0404391509541147,-0.16942110744293,0.117318819358521],
                   9:[-0.0107072484863877,-0.231376703354252,0.017541113036611,0.144871527682418,-0.041855459769806,0.0841832168332261,-0.0850020937282192,0.486170343825899,-0.0526717116822739,0.117318819358521],
                   10:[-0.0142238262314935,0.14752923682514,0.238235830732566,0.037950291904103,0.252075123381518,0.0474266904224567,-0.00363139069342027,0.274081442388563,-0.0599166970745255,-0.0527138812389185,0.117318819358521],
                   11:[-0.11837020183211,-0.635712481821264,0.239738832602538,0.330058936651707,-0.325784087988237,-0.0506514314589253,-0.281914404487009,0.852596345144291,0.651444614298805,-0.103476387303591,-0.354835880209975,0.117318819358521,],
                   12:[-0.00458164025442349,0.296219694015248,0.322146049419995,0.15917778285238,0.284864871688843,0.185509526463076,-0.0784621067883274,0.166312223692047,-0.284152486083397,-0.357125104338944,0.078437074055306,0.0884129667114481,0.117318819358521],
                   13:[-0.0545561913848106,0.675785423442753,0.423066443201941,-0.000165300126841193,0.104252994793763,-0.105763019303021,-0.15988308809318,0.0515050001032011,0.56013979290924,-0.45781539708603,-0.255870699752664,0.026960254296416,-0.0721245985053681,0.117318819358521],
                   14:[0.0649253995775223,-0.0216056457922249,-0.073738139377975,0.0931033310077225,-0.0194339577299149,-0.0879623837313009,0.057125517179467,0.205120850488097,0.132576503537441,0.489416890627328,-0.1106765720501,-0.081038793996096,0.0606031613503788,-0.00241467937442272,0.117318819358521],
                   15:[0.0459979286336779,0.0780075394482806,0.015021874148058,0.195180277284195,-0.00246643310153235,0.0473977117068314,-0.0682773558610363,0.19568019123878,-0.0876765449323747,0.177874852409192,-0.337519251582222,-0.0123255553640736,0.311573291192553,0.0458604327754991,0.278352222645651,0.117318819358521]}

        #coefficients for truncation error estimate
        _A1 = [0.0459979286336779,
               0.0780075394482806,
               0.015021874148058,
               0.195180277284195,
               -0.00246643310153235,
               0.0473977117068314,
               -0.0682773558610363,
               0.19568019123878,
               -0.0876765449323747,
               0.177874852409192,
               -0.337519251582222,
               -0.0123255553640736,
               0.311573291192553,
               0.0458604327754991,
               0.278352222645651,
               0.117318819358521]
        _A2 = [0.0603373529853206,
               0.175453809423998,
               0.0537707777611352,
               0.195309248607308,
               0.0135893741970232,
               -0.0221160259296707,
               -0.00726526156430691,
               0.102961059369124,
               0.000900215457460583,
               0.0547959465692338,
               -0.334995726863153,
               0.0464409662093384,
               0.301388101652194,
               0.00524851570622031,
               0.229538601845236,
               0.124643044573514]
        self.TR = [_a1 - _a2 for _a1, _a2 in zip(_A1, _A2)]


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
        timestep_rescale = 0.9 * (error_ratio)**(1/6)        

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #first stage is explicit
        if self.stage == 0:
            return 0.0
            
        #update timestep weighted slope 
        self.Ks[self.stage] = self.func(self.x, u, t)

        #compute slope and update fixed-point equation
        slope = sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage]))

        #use the jacobian
        if self.jac is not None:

            #most recent butcher coefficient
            b = self.BT[self.stage][self.stage]
            
            #compute jacobian of fixed-point equation
            jac_g = dt * b * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        performs the timestep update
        """

        #first stage is explicit
        if self.stage == 0:
            self.Ks[self.stage] = self.func(self.x, u, t)

        #restart anderson accelerator 
        self.acc.reset()    

        #error and step size control
        if self.stage < 15:
            self.stage += 1
            return True, 0.0, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)

        