from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from orca.geometry.input_parameters import InputParameterIterator
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.feature_transform import FeatureTransformPipeline
import numpy as np
import torch.nn as nn


@dataclass
class BaseGeometry(ABC):
    name: str
    stackup_xml: str
    simconfig_filename: str
    dataset: BaseDataset
    model: nn.Module
    input_parameter_iterator: InputParameterIterator
    features: FeatureTransformPipeline | None = None

    @property
    def input_iterator(self) -> InputParameterIterator | None:
        # Return the input parameter iterator, ensuring it is initialized with iter()
        return iter(self.input_parameter_iterator)

    @staticmethod
    @abstractmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        """
        Creates a GDS file based on the current input parameters.
        Input parameters are a list of values defining the geometry, e.g. width, length, radius etc.
        and will also be used as inputs for the AI/ML model.

        Returns:
            str: Path to the created GDS file.
        """

    def postprocess_outputs(
        self, output: dict[str, list], frequency_points: list | np.ndarray
    ) -> dict[str, list]:
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
