"""Base class and registry for ORCA surrogate models.

Every model in ORCA is an :class:`OrcaModel`. It sizes itself from the dataset's
:class:`~orca.training.spec.IOSpec` instead of hardcoded dimensions, declares the
frequency layout it consumes, declares which physical properties it guarantees by
construction, and owns its own hyperparameter search space and loss. That is what
makes architectures drag-and-drop replaceable: nothing outside the model class has
to change when you swap one for another.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import numpy as np
import skrf as rf
import torch
import torch.nn as nn

from orca.training.basis_expansion import IdentityBasis, BasisExpansion
from orca.training.guarantees import PhysicsGuarantees
from orca.training.spec import FrequencyMode, IOSpec


class OrcaModel(nn.Module, ABC):
    """Base class for all ORCA surrogate models.

    Subclasses must implement :meth:`from_spec`, :meth:`hyperparameter_search_space`
    and :meth:`forward`, and set :attr:`frequency_mode`.

    Every model accepts a :class:`~orca.training.basis_expansion.BasisExpansion`, so
    input expansions are shared across architectures instead of being reimplemented
    in each one. Two rules make a subclass basis-ready:

    1. size the first layer from :attr:`expanded_dim`, not from ``spec.input_dim``,
    2. start :meth:`forward` with ``x = self.basis(x)``.

    Doing it in ``forward`` rather than behind the scenes keeps the data path
    visible, and means the basis is traced into the exported ONNX graph like
    any other layer.

    Args:
        spec (IOSpec): The dataset contract this model was built against.
        basis (BasisExpansion | None): Fixed expansion applied to the inputs.
            Defaults to :class:`~orca.training.basis_expansion.IdentityBasis`,
            which changes nothing.
    """

    #: Frequency layout this model consumes. Checked against the dataset at wiring time.
    frequency_mode: ClassVar[FrequencyMode]

    #: Physical properties the architecture enforces by construction.
    guarantees: ClassVar[PhysicsGuarantees] = PhysicsGuarantees()

    def __init__(self, spec: IOSpec, basis: BasisExpansion | None = None):
        super().__init__()
        self.spec = spec
        self.basis = basis if basis is not None else IdentityBasis()

    @property
    def expanded_dim(self) -> int:
        """Width of the tensor the first layer sees, after the basis expansion."""
        return self.basis.expanded_dim(self.spec.input_dim)

    @classmethod
    @abstractmethod
    def from_spec(
        cls,
        spec: IOSpec,
        hyperparameters: dict[str, Any],
        basis: BasisExpansion | None = None,
    ) -> "OrcaModel":
        """Build a model sized for ``spec`` and configured by ``hyperparameters``.

        Implementations must tolerate extra keys in ``hyperparameters`` (the
        trainer and the basis expansion pass their own keys, such as ``learning_rate``
        or ``basis_degree``, through the same dictionary) and should fall back to
        sensible defaults for missing ones. ``basis`` is forwarded to
        :meth:`__init__` unchanged.
        """

    @staticmethod
    @abstractmethod
    def hyperparameter_search_space() -> dict[str, Any]:
        """Architecture hyperparameters for optuna tuning.

        Training hyperparameters (learning rate, batch size, epochs) belong to the
        trainer and are merged in separately; do not declare them here.
        """

    def default_loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Loss to train this model with when the trainer is not given one."""
        return nn.L1Loss()

    @classmethod
    def accepts_frequency_mode(cls, mode: FrequencyMode) -> bool:
        """Whether this model can consume samples laid out in ``mode``."""
        return mode is cls.frequency_mode

    def to_network(self, raw: np.ndarray, frequencies: np.ndarray) -> rf.Network:
        """Turn denormalized model outputs into a ``skrf.Network`` via the codec."""
        return self.spec.codec.to_network(raw, frequencies)


_MODEL_REGISTRY: dict[str, type[OrcaModel]] = {}


def register_model(name: str) -> Callable[[type[OrcaModel]], type[OrcaModel]]:
    """Class decorator registering a model under a short name.

    Args:
        name (str): Name to register the model under, e.g. ``"mlp"``.
    """

    def decorator(cls: type[OrcaModel]) -> type[OrcaModel]:
        if name in _MODEL_REGISTRY and _MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"Model name '{name}' is already registered.")
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(model: str | type[OrcaModel]) -> type[OrcaModel]:
    """Resolve a registered model name (or pass a model class straight through)."""
    if isinstance(model, type) and issubclass(model, OrcaModel):
        return model
    if isinstance(model, str):
        try:
            return _MODEL_REGISTRY[model]
        except KeyError:
            raise ValueError(
                f"Unknown model '{model}'. Available models: {available_models()}"
            ) from None
    raise TypeError(f"Expected an OrcaModel subclass or a registered name, got {model!r}.")


def available_models() -> list[str]:
    """Names of all registered models."""
    return sorted(_MODEL_REGISTRY)
