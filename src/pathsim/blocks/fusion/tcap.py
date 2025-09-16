#########################################################################################
##
##            Blocks for Thermal Cycle Absorption Process (TCAP) modelling
##                              (blocks/fusion/tcap.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from ..ode import ODE


# BLOCKS ================================================================================

class TCAP1D(ODE):
    """This block models the Thermal Cycle Absorption Process (TCAP) in 1d. 

    The model uses a 1d finite difference spatial discretization to construct 
    a nonlinear ODE internally as proposed in 

        https://doi.org/10.1016/j.ijhydene.2023.03.101


    """
    raise NotImplementedError("TCAP1D block is currently not impolemented!")


