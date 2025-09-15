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

class Bubbler(ODE):
    """
    Tritium bubbling system with 4 vials.

    Parameters
    ----------
    conversion_efficiency : float 
       Conversion efficiency from insoluble to soluble (between 0 and 1).
    vial_efficiency : float 
       collection efficiency of each vial (between 0 and 1).
    replacement_times : float | list[float] | list[list[float]] 
        times at which each vial is replaced. If None, no replacement events 
        are created. If a single value is provided, it is used for all vials.
          If a single list of floats is provided, it will be used for all vials.
        If a list of lists is provided, each sublist corresponds to the 
        replacement times for each vial.
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

        #create vial reset events
        self.events = self.create_reset_events()


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

        sol, ins = self.inputs.to_array()
        sample_out = (1 - ce) * ins + (1 - ve)**2 * (ce * ins + (1 - ve)**2 * sol)
        x = self.engine.get()

        y = np.hstack([x, sample_out])

        self.outputs.update_from_array(y)


    def _create_reset_event_vial(self, i, reset_times):

        def reset_vial_i(_):
            x = self.engine.get()
            x[i] = 0.0
            self.engine.set(x)

        return ScheduleList(
            times_evt=reset_times, 
            func_act=reset_vial_i
            )


    def create_reset_events(self):
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
        events = []

        # if reset_times is a single list use it for all vials
        if replacement_times is None:
            return events

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

        for i, ts in enumerate(replacement_times):
            events.append(self._create_reset_event_vial(i, ts))

        return events