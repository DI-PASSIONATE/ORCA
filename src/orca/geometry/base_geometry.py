from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from orca.geometry.input_parameters import InputParameterIterator
from orca.training.datasets.base_dataset import BaseDataset


@dataclass
class BaseGeometry(ABC):
    """
    Physical definition of a geometry class: how to draw it, which stackup and
    simulation settings it uses, and which parameters may vary.
    """

    name: str
    stackup_xml: str
    simconfig_filename: str
    dataset: BaseDataset
    input_parameter_iterator: InputParameterIterator

    @property
    def input_iterator(self) -> InputParameterIterator:
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
