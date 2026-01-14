from abc import ABC, abstractmethod
from orca.geometry.input_parameters import InputParameterIterator
from orca.logger import logger
import numpy as np

class BaseGeometry(ABC):
    def __init__(self, n_samples: int, name: str, stackup_xml: str, simconfig_filename: str, params: dict[str, any]|None = None):
        """
        Base class for defining geometries in ORCA.
        Geometries define the physical structure to be simulated, along with their input parameters and how to create GDS files.

        Args:
            name (str): Name of the geometry. Usually contains an ID or version number to distinguish differently parameterized instances.
            stackup_xml (str): Path to the stackup XML file defining the layer stackup for the geometry.
            simconfig_filename (str): Path to the simulation configuration file (.simcfg) used for Palace simulations.
            params (dict[str, any]|None): Optional dictionary of input parameters for the geometry instance. Used for logging and tracking purposes.
        """

        self._name = name
        self._n_samples = n_samples
        self._stackup_xml = stackup_xml
        self._simconfig_filename = simconfig_filename
        self._n_inputs = 0
        self._n_outputs = 0
        self.n_gds_generated = 0
        self._params = params

        if params is None: # If no params provided, we assume instance is a GeometryFactory
            self.input_iterator = iter(self.get_input_parameters())
            logger.info(f"Created GeometryFactory with input parameter iterator. Values: {self.input_iterator.input_values}")

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
    def params(self) -> dict[str, any]|None:
        return self._params
    
    @property
    def n_samples(self) -> int:
        return self._n_samples

    def create_geometry_instance(self, name: str, params: dict[str, any]) -> 'BaseGeometry':
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
            n_samples=1,
            name=name,
            stackup_xml=self.stackup_xml,
            simconfig_filename=self.simconfig_filename,
            params=params
        )
        self.n_gds_generated += 1
        return new_geometry

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
    def get_input_parameters(self) -> InputParameterIterator:
        """
        Generates the next set of input parameters for the geometry.
        This method can be used to iterate through different configurations of the geometry.

        Args:
            id (int): Unique identifier for the geometry instance. Corresponds to the amount of geometries generated so far.

        Returns:
            InputParameters: Next set of input parameters.
        """

    @abstractmethod
    def create_model(self):
        """
        Creates and returns a neural network model suitable for this geometry.
        The model architecture should be defined based on the specific input and output requirements of the geometry.

        Returns:
            nn.Module: The neural network model.
        """

    @abstractmethod
    def get_dataset(self):
        """
        Returns a dataset object suitable for training the model associated with this geometry.
        The dataset should provide input-output pairs relevant to the geometry's simulation tasks.

        Returns:
            torch.utils.data.Dataset: The dataset for training.
        """
        
    @abstractmethod
    def create_gds_file(self, params: dict[str, any]) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
            params: dict[str, any]: Dictionary of input parameters used to create the GDS file. Mapping is the same as what get_input_parameters returns.
        """

    def postprocess_outputs(self, output: dict[str, list], frequency_points: list|np.ndarray) -> dict[str, list]:
        """
        Postprocess the outputs of the ONNX model after inference.
        This method can be overridden by subclasses to apply any necessary transformations
        to the raw outputs produced by the ONNX model.

        Args:
            output (dict[str, list]): Dictionary of raw outputs from the ONNX model (key: output name, value: list of outputs)
            frequency_points (list|np.ndarray): Frequency points corresponding to each value in the output lists, if applicable.
        Returns:
            dict[str, any]: Postprocessed outputs. Default implementation returns the outputs unchanged.
        """
        return output