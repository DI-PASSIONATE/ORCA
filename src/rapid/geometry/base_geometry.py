import gdsfactory as gf

class BaseGeometry:
    def __init__(self, name: str, stackup_xml: str, simconfig_filename: str):
        self._name = name
        self._stackup_xml = stackup_xml
        self._simconfig_filename = simconfig_filename
        self._n_inputs = 0
        self._n_outputs = 0
        self._input_parameters = None  # To be defined by calling create_geometry

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
    
    @n_inputs.setter
    def n_inputs(self, value: int):
        self._n_inputs = value

    @property
    def n_outputs(self) -> int:
        return self._n_outputs
    
    @n_outputs.setter
    def n_outputs(self, value: int):
        self._n_outputs = value

    def create_geometry_instance(self, name: str, input_parameters: list = None) -> 'BaseGeometry':
        """
        Creates a new geometry instance based on the provided input parameters and name.
        This is useful when generating multiple geometry instances for data generation, e.g. 1000 transformers with different parameters.

        Args:
            input_parameters (list): List of input parameters defining the geometry.
            name (str): Name for the generated geometry instance.
        Returns:
            A new object extending BaseGeometry representing the created geometry.
        """
        assert len(input_parameters) == self.n_inputs if self.n_inputs > 0 else True, f"Expected {self.n_inputs} input parameters, got {len(input_parameters)}"

        # Create a new instance of the same class as self
        new_geometry = self.__class__(
            name=name,
            stackup_xml=self.stackup_xml,
            simconfig_filename=self.simconfig_filename
        )
        return new_geometry
    
    def create_gds_file(self) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
        """
        assert self._input_parameters is not None if self.n_inputs > 0 else True, "Input parameters must be set before creating GDS file."
        # Implement geometry creation logic here using gdsfactory
        # This is a placeholder implementation
    
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
        assert len(simulation_output) == self.n_outputs, f"Expected {self.n_outputs} output values, got {len(simulation_output)}"

    def get_next_input_parameters(self) -> list:
        """
        Generates the next set of input parameters for the geometry.
        This method can be used to iterate through different configurations of the geometry.

        Returns:
            list: Next set of input parameters.
        """
        raise NotImplementedError("Subclasses must implement get_next_input_parameters method.")