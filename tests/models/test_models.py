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


def for_each_model():
    """
    Decorator to create test methods for each file matching the pattern.
    """
    def decorator(test_class):
        #get the path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "_models")
        
        def create_test(file_path):
            def test(self):
                
                #load simulation duration from metadata
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    metadata = data["metadata"]
                    duration = metadata.get("duration", 1.0)
                
                #load the simulation from file
                sim = Simulation.load(file_path, log=False)
            
                #run the simulation for duration from metadata
                sim.run(duration)     

            return test
        
        #create a test method for each file matching pattern
        for file_path in glob.glob(os.path.join(models_dir, "*.mdl")):
            model_name = os.path.basename(file_path).replace('.', '_')
            test_name = f"test_{model_name}"
            setattr(test_class, test_name, create_test(file_path))
                    
        return test_class
    
    return decorator


@for_each_model()
class TestModels(unittest.TestCase):
    """
    load and run simulation models
    """
    pass


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)