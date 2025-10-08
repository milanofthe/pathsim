#########################################################################################
##
##                           FUNCTIONAL MOCK-UP UNIT (FMU) BLOCKS
##                                   (pathsim/blocks/fmu.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..events.schedule import Schedule 
from ..optim.operator import DynamicOperator


# BLOCKS ================================================================================

class CoSimulationFMU(Block):
    """Co-Simulation FMU block using FMPy.

    This block wraps an FMU (Functional Mock-up Unit) for co-simulation. 
    The FMU encapsulates a simulation model that can be executed independently 
    and synchronized with the main simulation.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    start_values : dict, optional
        dictionary of variable names and their initial values
    dt : float, optional
        communication step size for co-simulation, default: None

    Attributes
    ----------
    fmu_path : str
        path to the FMU file
    instance_name : str
        name of the FMU instance
    start_values : dict
        initial values for FMU variables
    model_description : ModelDescription
        FMI model description from FMPy
    fmu : FMU2Slave
        FMPy FMU instance for co-simulation
    _input_refs : dict
        reference map for input variables
    _output_refs : dict
        reference map for output variables
    """

    #max number of ports (will be configured based on FMU)
    _n_in_max = None
    _n_out_max = None

    #maps for input and output port labels
    _port_map_in = {}
    _port_map_out = {}

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None, dt=None):

        # Import FMPy here to avoid requiring it as a dependency if not used
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave
        except ImportError:
            raise ImportError("FMPy is required for FMU blocks. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.start_values = start_values if start_values is not None else {}
        self.dt = dt

        # Read model description
        self.model_description = read_model_description(fmu_path)

        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Get input and output variable references
        self._input_refs = {}
        self._output_refs = {}

        for variable in self.model_description.modelVariables:
            if variable.causality == 'input':
                self._input_refs[variable.name] = variable.valueReference
                self._port_map_in[variable.name] = len(self._input_refs) - 1
            elif variable.causality == 'output':
                self._output_refs[variable.name] = variable.valueReference
                self._port_map_out[variable.name] = len(self._output_refs) - 1

        # Initialize base class with proper port configuration
        super().__init__()

        # Instantiate FMU
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName=self.instance_name
            )

        # Setup experiment
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)

        # Set start values
        for name, value in self.start_values.items():
            for variable in self.model_description.modelVariables:
                if variable.name == name:
                    if variable.type == 'Real':
                        self.fmu.setReal([variable.valueReference], [value])
                    elif variable.type == 'Integer':
                        self.fmu.setInteger([variable.valueReference], [int(value)])
                    elif variable.type == 'Boolean':
                        self.fmu.setBoolean([variable.valueReference], [bool(value)])

        # Enter initialization mode
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        # Internal scheduled event function
        self.events = [
            Schedule(
                t_start=0,
                t_period=self.dt,
                func_act=self._step_fmu
                )
            ]

        # Read initial outputs
        self._update_outputs_from_fmu()


    def _step_fmu(self, t):
        """Perform one FMU co-simulation step"""
        self._update_fmu_from_inputs()    

        # Perform co-simulation step
        self.fmu.doStep(
            currentCommunicationPoint=t, 
            communicationStepSize=self.dt
            )  

        self._update_outputs_from_fmu()


    def _update_fmu_from_inputs(self):
        """Read block inputs and update FMU outputs."""
        if len(self._input_refs) > 0:
            input_values = self.inputs.to_array()
            input_vrefs = list(self._input_refs.values())
            self.fmu.setReal(input_vrefs, input_values.tolist())


    def _update_outputs_from_fmu(self):
        """Read outputs from FMU and update block outputs."""
        if len(self._output_refs) > 0:
            output_vrefs = list(self._output_refs.values())
            output_values = self.fmu.getReal(output_vrefs)
            self.outputs.update_from_array(np.array(output_values))


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu.reset()
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self._update_outputs_from_fmu()


    def __len__(self):
        """FMU is a discrete time source-like block without direct passthrough"""
        return 0


    def __del__(self):
        """Cleanup FMU resources."""
        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
        except:
            pass


#future
class ModelExchangeFMU(Block):
    def __init__():
        raise NotImplementedError()