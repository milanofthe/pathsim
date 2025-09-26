#########################################################################################
##
##                                      Bubbler Block
##                                (blocks/fusion/bubbler.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ..ode import ODE
from ...events.schedule import ScheduleList


# BLOCK DEFIINITIONS ====================================================================

class Bubbler4(ODE):
    """
    Tritium bubbling system with sequential vial collection stages.

    This block models a tritium collection system used in fusion reactor blanket 
    purge gas processing. The system bubbles tritium-containing gas through a series 
    of liquid-filled vials to capture and concentrate tritium for measurement and 
    inventory tracking.


    Physical Description
    --------------------
    The bubbler consists of two parallel processing chains:

    **Soluble Chain (Vials 1-2):** 
    Tritium already in soluble forms (HTO, HT) flows sequentially through 
    vials 1 and 2. Each vial has a collection efficiency :math:`\\eta_{vial}`, 
    representing the fraction of tritium that dissolves into the liquid phase 
    and is retained.

    **Insoluble Chain (Vials 3-4):** 
    Tritium in insoluble forms (T₂, organically bound) first undergoes catalytic 
    conversion to soluble forms with efficiency :math:`\\alpha_{conv}`. The 
    converted tritium, along with uncaptured soluble tritium from the first chain, 
    then flows through vials 3 and 4 with the same collection efficiency.


    Mathematical Formulation
    -------------------------
    The system is governed by the following differential equations for the 
    vial inventories :math:`x_i`:

    .. math::

        \\frac{dx_1}{dt} &= \\eta_{vial} \\cdot u_{sol}

        \\frac{dx_2}{dt} &= \\eta_{vial} \\cdot (1-\\eta_{vial}) \\cdot u_{sol}

        \\frac{dx_3}{dt} &= \\eta_{vial} \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}]

        \\frac{dx_4}{dt} &= \\eta_{vial} \\cdot (1-\\eta_{vial}) \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}]


    The sample output represents uncaptured tritium exiting the system:

    .. math::
    
        y_{sample} = (1-\\alpha_{conv}) \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot [\\alpha_{conv} \\cdot u_{insol} + (1-\\eta_{vial})^2 \\cdot u_{sol}]


    Where:
        - :math:`u_{sol}` = soluble tritium input flow rate
        - :math:`u_{insol}` = insoluble tritium input flow rate  
        - :math:`\\eta_{vial}` = vial collection efficiency
        - :math:`\\alpha_{conv}` = conversion efficiency from insoluble to soluble
        - :math:`x_i` = tritium inventory in vial i

    Parameters
    ----------
    conversion_efficiency : float 
       Conversion efficiency from insoluble to soluble forms (:math:`\\alpha_{conv}`), 
       between 0 and 1.
    vial_efficiency : float 
       Collection efficiency of each vial (:math:`\\eta_{vial}`), between 0 and 1.
    replacement_times : float | list[float] | list[list[float]] 
        Times at which each vial is replaced with a fresh one. If None, no 
        replacement events are created. If a single value is provided, it is 
        used for all vials. If a single list of floats is provided, it will be 
        used for all vials. If a list of lists is provided, each sublist 
        corresponds to the replacement times for each vial.

    Notes
    -----
    Vial replacement is modeled as instantaneous reset events that set the 
    corresponding vial inventory to zero, simulating the physical replacement 
    of a full vial with an empty one.
    """

    _port_map_out = {
        "vial1": 0,
        "vial2": 1,
        "vial3": 2,
        "vial4": 3,
        "sample_out": 4,
    }
    _port_map_in = {
        "sample_in_soluble": 0,
        "sample_in_insoluble": 1,
    }

    def __init__(
        self,
        conversion_efficiency=0.9,
        vial_efficiency=0.9,
        replacement_times=None,
        ):

        #bubbler parameters
        self.replacement_times = replacement_times
        self.vial_efficiency = vial_efficiency
        self.conversion_efficiency = conversion_efficiency

        def _fn(x, u, t):

            #short
            ve = self.vial_efficiency
            ce = self.conversion_efficiency

            #unpack inputs
            sol, ins = u

            #compute vial content change rates
            dv1 = ve * sol
            dv2 = dv1 * (1 - ve)
            dv3 = ve * (ce * ins + (1 - ve)**2 * sol)
            dv4 = dv3 * (1 - ve)

            return np.array([dv1, dv2, dv3, dv4])

        super().__init__(func=_fn, initial_value=np.zeros(4))

        #create internal vial reset events
        self._create_reset_events()


    def update(self, t):
        """update global system equation

        Parameters
        ----------
        t : float
            evaluation time
        """

        #short
        ve = self.vial_efficiency
        ce = self.conversion_efficiency

        sol, ins = self.inputs[0], self.inputs[1]
        sample_out = (1 - ce) * ins + (1 - ve)**2 * (ce * ins + (1 - ve)**2 * sol)
        x = self.engine.get()

        y = np.hstack([x, sample_out])

        self.outputs.update_from_array(y)


    def _create_reset_event_vial(self, i, reset_times):
        """Define event action function and return a `ScheduleList` event 
        per vial `i` that triggers at predefined `reset_times`. 
        """

        def reset_vial_i(_):
            #get the full engine state
            x = self.engine.get()
            #set index 'i' to zero
            x[i] = 0.0
            #set the full engine state 
            self.engine.set(x)

        return ScheduleList(
            times_evt=reset_times, 
            func_act=reset_vial_i
            )


    def _create_reset_events(self):
        """Create reset events for all vials based on the replacement times.

        Raises
        ------
        ValueError : If reset_times is not valid.

        Returns
        -------
        events : list[ScheduleList]
            list of reset events for vials
        """

        replacement_times = self.replacement_times
        self.events = []

        # if reset_times is a single list use it for all vials
        if replacement_times is None:
            return 

        if isinstance(replacement_times, (int, float)):
            replacement_times = [replacement_times]

        # if it's a flat list use it for all vials
        elif isinstance(replacement_times, list) and all(
            isinstance(t, (int, float)) for t in replacement_times
            ):
            replacement_times = [replacement_times] * 4

        elif isinstance(replacement_times, np.ndarray) and replacement_times.ndim == 1:
            replacement_times = [replacement_times.tolist()] * 4

        elif isinstance(replacement_times, list) and len(replacement_times) != 4:
            raise ValueError(
                "replacement_times must be a single value or a list with the same length as the number of vials"
            )

        #create the internal events
        self.events = [
            self._create_reset_event_vial(i, ts) for i, ts in enumerate(replacement_times)
            ]
