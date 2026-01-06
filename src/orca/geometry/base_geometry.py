from abc import ABC, abstractmethod
from orca.geometry.input_parameters import InputParameters

class BaseGeometry(ABC):
    def __init__(self, name: str, stackup_xml: str, simconfig_filename: str):
        self._name = name
        self._stackup_xml = stackup_xml
        self._simconfig_filename = simconfig_filename
        self._n_inputs = 0
        self._n_outputs = 0
        self._input_parameters = None
        self.n_gds_generated = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def stackup_xml(self) -> str:
        return self._stackup_xml

    @property
    def simconfig_filename(self) -> str:
        return self._simconfig_filename

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def input_parameters(self) -> InputParameters|None:
        return self._input_parameters
            
    @n_inputs.setter
    def n_inputs(self, value: int):
        self._n_inputs = value

    @n_outputs.setter
    def n_outputs(self, value: int):
        self._n_outputs = value

    @input_parameters.setter
    def input_parameters(self, value: InputParameters):
        if value.n_inputs != self._n_inputs:
            raise ValueError(f"InputParameters n_inputs ({value.n_inputs}) does not match geometry n_inputs ({self._n_inputs})")  
        self._input_parameters = value

    def create_geometry_instance(self, name: str, input_parameters: InputParameters) -> 'BaseGeometry':
        """
        Creates a new geometry instance based on the provided input parameters and name.
        This is useful when generating multiple geometry instances for data generation, e.g. 1000 transformers with different parameters.

        Args:
            input_parameters (list): List of input parameters defining the geometry.
            name (str): Name for the generated geometry instance.
        Returns:
            A new object extending BaseGeometry representing the created geometry.
        """
        # Create a new instance of the same class as self
        new_geometry = self.__class__(
            name=name,
            stackup_xml=self.stackup_xml,
            simconfig_filename=self.simconfig_filename
        )
        new_geometry.input_parameters = input_parameters
        self.n_gds_generated += 1
        return new_geometry
    
    @abstractmethod
    def create_gds_file(self) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
        """

    def _create_gds_file(self) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
        """
        if self.n_inputs > 0 and self._input_parameters is None:
            raise ValueError("Input parameters must be set before creating GDS file.")
        
        print(self.input_parameters)
        return self.create_gds_file()

    #@abstractmethod
    def simulation_output_to_scalar(self, simulation_output: list) -> int:
        """
        Converts simulation output to a scalar value.
        This is used for calculating the loss function during optimization, to determine how well the geometry meets the design goals.
        Subclasses should implement this method based on their specific output requirements.

        Args:
            simulation_output (list): List of output values from the simulation, e.g. S-parameters.
        
        Returns:
            int: Scalar value representing the performance of the geometry.
        """

    @abstractmethod
    def get_next_input_parameters(self, idx: int) -> InputParameters:
        """
        Generates the next set of input parameters for the geometry.
        This method can be used to iterate through different configurations of the geometry.

        Args:
            id (int): Unique identifier for the geometry instance. Corresponds to the amount of geometries generated so far.

        Returns:
            InputParameters: Next set of input parameters.
        """