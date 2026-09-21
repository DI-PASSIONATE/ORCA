from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

    def is_feasible(self, params: dict[str, Any]) -> bool:  # noqa: ARG002 - hook with a default
        """
        Whether a parameter combination describes a layout that can be drawn.

        Called before :meth:`create_gds_file`, for every draw of the input
        parameter iterator; infeasible draws are rejected and redrawn, so the
        requested number of samples is met and only buildable layouts are
        simulated. Override this for cheap, closed-form constraints between
        parameters (a winding that must fit its diameter, a via array that must
        fit its trace). Do not clamp or repair parameters instead: the parameter
        table records the requested values, and a repaired layout would train
        the model on a geometry it does not have.

        The default accepts everything. :meth:`create_gds_file` should still
        raise ``ValueError`` for a combination it cannot draw; that is the
        safety net, this is the filter.
        """
        return True

    def feasibility_constraints(self) -> list[str]:
        """
        :meth:`is_feasible` as expressions a consumer of the model can evaluate.

        Each string is a boolean expression over the input parameter names in
        the grammar of :mod:`orca.geometry.constraints`, for example
        ``"bottom_linewidth <= bottom_winding_diameter / 3"``. The ONNX
        exporter writes them to the ``input_constraints`` metadata key, so
        COBRA can refuse a query for a geometry that cannot be built rather
        than return a prediction the model was never trained for.

        They must accept exactly the parameter sets :meth:`is_feasible`
        accepts; derive both from the same numbers. The default declares no
        constraints, which a consumer reads as "the whole parameter box".
        """
        return []

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
