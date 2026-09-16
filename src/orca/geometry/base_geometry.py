from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from orca.geometry.input_parameters import InputParameterIterator

if TYPE_CHECKING:
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
    input_parameter_iterator: InputParameterIterator

    @property
    def input_iterator(self) -> InputParameterIterator:
        # Return the input parameter iterator, ensuring it is initialized with iter()
        return iter(self.input_parameter_iterator)

    @cached_property
    def dataset(self) -> BaseDataset:
        """
        The dataset used to train a model of this geometry, built once per
        instance by :meth:`create_dataset`. Each geometry instance gets its own
        dataset, so the normalizer statistics fitted in one training run never
        leak into another.
        """
        return self.create_dataset()

    @cached_property
    def n_ports(self) -> int:
        """
        Number of ports of the Palace simulation, read from the ``ports`` list of
        the simulation config. This sets the ``.sNp`` extension of the Touchstone
        results and must match the codec of :attr:`dataset`.
        """
        from orca.simulation.simulate import read_simconfig

        return len(read_simconfig(self.simconfig_filename)["ports"])

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

    @abstractmethod
    def create_dataset(self) -> BaseDataset:
        """
        Builds the training dataset for this geometry: a :class:`BaseDataset`
        subclass with its output codec and normalizers. Called once per instance,
        the first time :attr:`dataset` is read.

        Import the ``orca.training`` modules inside this method rather than at
        the top of the geometry file, so the geometry can still be used for GDS
        generation and simulation when PyTorch is not installed.

        Returns:
            BaseDataset: A fresh, unfitted dataset.
        """
