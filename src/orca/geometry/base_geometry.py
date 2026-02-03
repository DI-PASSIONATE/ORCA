from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from orca.geometry.input_parameters import InputParameterIterator
from orca.logger import logger
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.feature_transform import FeatureTransformPipeline
import numpy as np
import torch.nn as nn

@dataclass
class BaseGeometry(ABC):
    n_samples: int
    name: str
    stackup_xml: str
    simconfig_filename: str
    dataset: BaseDataset
    model: nn.Module
    input_parameter_iterator: InputParameterIterator
    features: FeatureTransformPipeline|None = None

    def __post_init__(self):
        # Ensure the input parameter iterator knows the number of samples
        # This is done here instead of passing it to the constructor of InputParameterIterator
        # because dataclass fields are initialized statically.
        self.input_parameter_iterator.set_sample_count(self.n_samples)

    @property
    def input_iterator(self) -> InputParameterIterator | None:
        # Return the input parameter iterator, ensuring it is initialized with iter()
        return iter(self.input_parameter_iterator)
        
    @staticmethod
    @abstractmethod
    def create_gds_file(name: str, params: dict[str, Any]) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
            params: dict[str, Any]: Dictionary of input parameters used to create the GDS file. Mapping is the same as what get_input_parameters returns.
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