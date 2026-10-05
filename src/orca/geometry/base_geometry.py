from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import skrf as rf

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

    #: Electrical parameters whose test error is an absolute difference rather than a
    #: relative one, e.g. a coupling factor that is close to zero for weak coupling.
    absolute_error_parameters: ClassVar[frozenset[str]] = frozenset()

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
        return len(self._simconfig_ports)

    @cached_property
    def _simconfig_ports(self) -> list[dict[str, Any]]:
        from orca.simulation.simulate import read_simconfig

        return read_simconfig(self.simconfig_filename)["ports"]

    def simulation_ports(self, params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG002 - hook with a default
        """
        The gds2palace ports of one sample, as entries of the simconfig's ``ports`` list.

        Which metal a port has to reach is a property of the layout, so it may depend
        on the parameters: a single-turn inductor feeds on another metal than a
        multi-turn one. Override this to change port fields per sample, typically
        ``to_layername``; the simulation settings stay those of the simconfig, so
        every sample of one model is meshed and solved alike. Keep every port, with
        its number and in its order: the Touchstone files and the model's outputs
        depend on them, and :meth:`ports_for` checks it.

        The default returns the simconfig's ports unchanged.

        Args:
            params (dict[str, Any]): The sample's input parameters, as in the
                parameter table (integers may come back as floats).

        Returns:
            list[dict[str, Any]]: A fresh copy of the port entries, safe to modify.
        """
        return copy.deepcopy(self._simconfig_ports)

    def ports_for(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """
        :meth:`simulation_ports` for one sample, checked against the simconfig.

        Raises:
            ValueError: The ports are not the simconfig's port numbers in its order.
        """
        ports = self.simulation_ports(params)
        expected = [port["portnumber"] for port in self._simconfig_ports]
        numbers = [port.get("portnumber") for port in ports]
        if numbers != expected:
            raise ValueError(
                f"{type(self).__name__}.simulation_ports returned ports {numbers} for "
                f"{params}; the simconfig defines {expected}, and every sample must keep "
                "those numbers in that order."
            )
        return ports

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

    def electrical_parameters(self, ntwk: rf.Network) -> dict[str, np.ndarray]:  # noqa: ARG002 - hook with a default
        """
        The figures of merit of a simulated or predicted network, which ``ModelTester``
        reports the model's error for, next to the S-parameter error.

        Which port is which is a property of the layout, so the geometry decides how to
        read its network: for example the differential L, R and Q of an inductor whose
        center tap is AC-grounded. :mod:`orca.utils.postprocessing` has the usual ones
        (:func:`~orca.utils.postprocessing.inductor_parameters`,
        :func:`~orca.utils.postprocessing.transformer_parameters`). Import it inside
        this method, as the presets do.

        The default returns nothing, so only the S-parameter error is reported.

        Args:
            ntwk (rf.Network): The network, simulated or predicted.

        Returns:
            dict: Parameter name to a curve over ``ntwk.f`` or to a scalar (a 0-d
            array), such as a self-resonance frequency. Errors are relative unless the
            name is in :attr:`absolute_error_parameters`.
        """
        return {}

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
