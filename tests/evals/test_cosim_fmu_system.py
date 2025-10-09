########################################################################################
##
##                         Testing System with Co-Simulation FMU
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os

from pathlib import Path
TEST_DIR = Path(__file__).parent

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Source, CoSimulationFMU



# TESTCASE =============================================================================

@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestCoSimFMUSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        fmu_path = TEST_DIR / r"CoupledClutches_CS.fmu"

        #blocks that define the system
        self.fmu = CoSimulationFMU(fmu_path, dt=0.01)
        src = Source(lambda t: 0.1 * np.sin(5*t))
        self.sco = Scope()

        #initialize simulation with the blocks, connections
        self.sim = Simulation(
            blocks=[self.fmu, self.sco, src],
            connections=[
                Connection(src[0], self.fmu[0], self.sco[4]),
                Connection(self.fmu[:4], self.sco[:4])
                ],
            dt=self.fmu.dt/5,
            log=False
            )


    def test_eval(self):

        self.sim.run(20)

        time, [a, b, c, d, e] = self.sco.read()

        #check number of fmu steps
        self.assertEqual(len(self.fmu.events[0]), 2001)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
