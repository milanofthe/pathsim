########################################################################################
##
##                             TEST ALL AVAILABLE MODELS
##
##                                 Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os
import glob
import json

from pathsim import Simulation


# TESTS ================================================================================

def test_for_each_model(file_pattern):

    """
    Decorator to create test methods for each file matching the pattern.
    """

    def decorator(test_class):
            
        def create_test(file_path):

            def test(self):

                #load simulation duration from metadata
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    duration = data["metadata"].get("duration", 1.0)

                #load the simulation from file
                sim = Simulation.load(file_path, log=False)

                sim.run(duration)

            return test
        
        #ceate a test method for each file matching pattern
        for file_path in glob.glob(file_pattern):
            model_name = os.path.basename(file_path)
            test_name = f"test_{model_name}"
            setattr(test_class, test_name, create_test(file_path))
                    
        return test_class
    
    return decorator


@test_for_each_model("_models/*.mdl")
class TestModels(unittest.TestCase):
    """
    load and run simulation models
    """
    pass


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)