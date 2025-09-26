########################################################################################
##
##                                  TESTS FOR 
##                         'blocks.tritium.bubbler.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.tritium import Bubbler4

from pathsim.solvers import EUF
from pathsim.events.schedule import ScheduleList


# TESTS ================================================================================

class TestFusionBubbler4(unittest.TestCase):
    """
    Test the implementation of the 'Bubbler4' block class from the fusion toolbox.
    The block inherits from `ODE` and models a 4-vial tritium collection system.
    """
    
    def test_init(self):
        """Test initialization with various parameter combinations"""
        
        # Default initialization
        B = Bubbler4()
        self.assertEqual(B.conversion_efficiency, 0.9)
        self.assertEqual(B.vial_efficiency, 0.9)
        self.assertEqual(B.replacement_times, None)
        self.assertEqual(len(B.events), 0)  # No events when replacement_times is None
        
        # Specific initialization
        B = Bubbler4(conversion_efficiency=0.8, vial_efficiency=0.7, replacement_times=[100, 200])
        self.assertEqual(B.conversion_efficiency, 0.8)
        self.assertEqual(B.vial_efficiency, 0.7)
        self.assertEqual(B.replacement_times, [100, 200])
        
        # Set solver to check internal solver instance
        B.set_solver(EUF, parent=None)
        self.assertTrue(B.engine)
        np.testing.assert_array_equal(B.engine.initial_value, np.zeros(4))
    

    def test_init_replacement_times(self):
        """Test different replacement_times configurations"""
        
        # Single list - should replicate for all vials
        B = Bubbler4(replacement_times=[100, 200, 300])
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)
        
        # List of lists - one per vial
        times = [[100, 200], [150, 250], [120, 220], [180, 280]]
        B = Bubbler4(replacement_times=times)
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)
        for event in B.events:
            self.assertIsInstance(event, ScheduleList)
        
        # NumPy array
        B = Bubbler4(replacement_times=np.array([100, 200, 300]))
        B.set_solver(EUF, parent=None)
        self.assertEqual(len(B.events), 4)
        
        # Invalid case - wrong length
        with self.assertRaises(ValueError):
            B = Bubbler4(replacement_times=[[100], [200], [300]])  # Only 3 vials
    

    def test_update_outputs(self):
        """Test the update method and output calculations"""
        
        # Test with default parameters
        B = Bubbler4()
        B.set_solver(EUF, parent=None)
        
        # Set some test inputs
        B.inputs[0] = 10.0  # soluble input
        B.inputs[1] = 5.0   # insoluble input
        
        B.update(0.0)
        
        # Check that outputs are set (initial state is zero)
        self.assertEqual(B.outputs[0], 0.0)  # vial 1
        self.assertEqual(B.outputs[1], 0.0)  # vial 2
        self.assertEqual(B.outputs[2], 0.0)  # vial 3
        self.assertEqual(B.outputs[3], 0.0)  # vial 4
        
        # Calculate expected sample_out with default parameters (ve=0.9, ce=0.9)
        ve, ce = 0.9, 0.9
        sol, ins = 10.0, 5.0
        expected_sample_out = (1-ce)*ins + (1-ve)**2 * (ce*ins + (1-ve)**2 * sol)
        self.assertAlmostEqual(B.outputs[4], expected_sample_out, places=10)


    def test_mass_conservation(self):
        """Test that mass is conserved in the system"""
        
        B = Bubbler4(conversion_efficiency=0.8, vial_efficiency=0.7)
        B.set_solver(EUF, parent=None)
        
        # Set test inputs
        sol_in, ins_in = 10.0, 5.0
        B.inputs[0] = sol_in
        B.inputs[1] = ins_in
        
        # Get rates and sample_out
        u = np.array([sol_in, ins_in])
        x = np.zeros(4)
        rates = B.func(x, u, 0.0)
        
        B.update(0.0)
        sample_out = B.outputs[4]
        
        # Total input rate
        total_in = sol_in + ins_in
        
        # Total accumulation rate in vials
        total_vial_rates = np.sum(rates)
        
        # Mass conservation: input = vial accumulation + sample output
        total_out = total_vial_rates + sample_out
        
        self.assertAlmostEqual(total_in, total_out, places=10)
    

    def test_vial_reset_functionality(self):
        """Test that vial reset events work correctly"""
        
        B = Bubbler4(replacement_times=[10.0])
        B.set_solver(EUF, parent=None)
        
        # Manually set some vial inventories
        x = np.array([1.0, 2.0, 3.0, 4.0])
        B.engine.set(x)
        
        # Test reset function for vial 0
        reset_event = B.events[0]
        reset_func = reset_event.func_act
        
        # Execute reset
        reset_func(None)
        
        # Check that vial 0 was reset but others remain
        x_after = B.engine.get()
        self.assertEqual(x_after[0], 0.0)
        self.assertEqual(x_after[1], 2.0)
        self.assertEqual(x_after[2], 3.0)
        self.assertEqual(x_after[3], 4.0)

        # Test reset function for vial 3
        reset_event = B.events[3]
        reset_func = reset_event.func_act
        
        # Execute reset
        reset_func(None)
        
        # Check that vial 0 was reset but others remain
        x_after = B.engine.get()
        self.assertEqual(x_after[0], 0.0)
        self.assertEqual(x_after[1], 2.0)
        self.assertEqual(x_after[2], 3.0)
        self.assertEqual(x_after[3], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)