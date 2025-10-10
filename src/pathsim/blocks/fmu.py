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
    """Co-Simulation FMU block using FMPy with support for FMI 2.0 and FMI 3.0.

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
        communication step size for co-simulation. If None, uses the FMU's 
        default experiment step size if available.

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
    fmu : FMU2Slave or FMU3Slave
        FMPy FMU instance for co-simulation
    fmi_version : str
        FMI version ('2.0' or '3.0')
    _input_refs : dict
        reference map for input variables
    _output_refs : dict
        reference map for output variables
    
    FMU Capabilities
    ----------------
    can_interpolate_inputs : bool
        whether FMU can interpolate inputs between communication points
    can_handle_variable_step : bool
        whether FMU supports variable communication step sizes
    default_step_size : float or None
        recommended default step size from FMU
    max_output_derivative_order : int
        maximum order of output derivatives available
    """

    #max number of ports (will be configured based on FMU)
    _n_in_max = None
    _n_out_max = None

    #maps for input and output port labels
    _port_map_in = {}
    _port_map_out = {}

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None, 
                 dt=None, verbose=False):

        # Import FMPy here to avoid requiring it as a dependency if not used
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave
            from fmpy.fmi3 import FMU3Slave
        except ImportError:
            raise ImportError("FMPy is required for FMU blocks. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.start_values = start_values if start_values is not None else {}
        self.verbose = verbose

        # Read model description
        self.model_description = read_model_description(fmu_path)
        
        # Detect FMI version
        self.fmi_version = self.model_description.fmiVersion
        
        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Extract metadata and capabilities
        self._extract_fmu_metadata()
        
        # Use provided dt or fall back to FMU default
        if dt is None:
            if self.default_step_size is not None:
                self.dt = self.default_step_size
            else:
                raise ValueError("No step size provided and FMU has no default experiment step size")
        else:
            self.dt = dt

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

        # Instantiate FMU based on version
        if self.fmi_version.startswith('2.'):
            self.fmu = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName=self.instance_name
            )
        elif self.fmi_version.startswith('3.'):
            self.fmu = FMU3Slave(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName=self.instance_name
            )
        else:
            raise ValueError(f"Unsupported FMI version: {self.fmi_version}")

        # Setup experiment
        self.fmu.instantiate()
        
        # FMI 3.0 has different initialization sequence
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=None,
                startTime=0.0,
                stopTime=None
            )
        else:
            self.fmu.setupExperiment(startTime=0.0)
            self.fmu.enterInitializationMode()

        # Set start values
        self._set_start_values()

        # Exit initialization mode
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
        

    def _extract_fmu_metadata(self):
        """Extract metadata and capabilities from FMU."""
        
        cs = self.model_description.coSimulation
        
        if cs is None:
            raise ValueError("FMU does not support Co-Simulation")
        
        # Extract capabilities (consistent across FMI 2.0 and 3.0)
        self.can_interpolate_inputs = getattr(cs, 'canInterpolateInputs', False)
        self.can_handle_variable_step = getattr(cs, 'canHandleVariableCommunicationStepSize', False)
        self.max_output_derivative_order = getattr(cs, 'maxOutputDerivativeOrder', 0)
        
        # FMI 3.0 specific capabilities
        if self.fmi_version.startswith('3.'):
            self.can_return_early = getattr(cs, 'canReturnEarlyAfterIntermediateUpdate', False)
            self.has_event_mode = getattr(cs, 'hasEventMode', False)
            self.recommended_intermediate_input_smoothness = getattr(cs, 'recommendedIntermediateInputSmoothness', 0)
        else:
            self.can_return_early = False
            self.has_event_mode = False
            self.recommended_intermediate_input_smoothness = 0
        
        # Extract default experiment settings
        default_experiment = self.model_description.defaultExperiment
        
        if default_experiment is not None:
            self.default_start_time = getattr(default_experiment, 'startTime', 0.0)
            self.default_stop_time = getattr(default_experiment, 'stopTime', None)
            self.default_step_size = getattr(default_experiment, 'stepSize', None)
            self.default_tolerance = getattr(default_experiment, 'tolerance', None)
        else:
            self.default_start_time = 0.0
            self.default_stop_time = None
            self.default_step_size = None
            self.default_tolerance = None
        
        # Model metadata
        self.model_name = self.model_description.modelName
        self.generation_tool = getattr(self.model_description, 'generationTool', 'Unknown')
        self.generation_date = getattr(self.model_description, 'generationDateAndTime', 'Unknown')
        self.description = getattr(self.model_description, 'description', '')
        self.author = getattr(self.model_description, 'author', 'Unknown')
        self.version = getattr(self.model_description, 'version', 'Unknown')


    def _set_start_values(self):
        """Set initial values for FMU variables."""
        for name, value in self.start_values.items():
            for variable in self.model_description.modelVariables:
                if variable.name == name:
                    if variable.type in ['Real', 'Float64', 'Float32']:  # FMI 3.0 uses Float64/Float32
                        self.fmu.setReal([variable.valueReference], [value])
                    elif variable.type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
                        self.fmu.setInteger([variable.valueReference], [int(value)])
                    elif variable.type == 'Boolean':
                        self.fmu.setBoolean([variable.valueReference], [bool(value)])


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
        """Read block inputs and update FMU inputs."""
        if len(self._input_refs) > 0:
            input_vrefs = list(self._input_refs.values())
            self.fmu.setReal(input_vrefs, self.inputs.to_array())


    def _update_outputs_from_fmu(self):
        """Read outputs from FMU and update block outputs."""
        if len(self._output_refs) > 0:
            output_vrefs = list(self._output_refs.values())
            self.outputs.update_from_array(self.fmu.getReal(output_vrefs))


    def update(self, t):
        """Update FMU inputs/outputs between scheduled steps if interpolation supported."""
        if self.can_interpolate_inputs:
            self._update_fmu_from_inputs()
            self._update_outputs_from_fmu()


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu.reset()
        
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=None,
                startTime=0.0,
                stopTime=None
            )
        else:
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