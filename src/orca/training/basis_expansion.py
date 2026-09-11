"""Basis expansions of model inputs, shared by every ORCA architecture.

A :class:`BasisExpansion` widens a model's input tensor before its first layer by
appending fixed basis functions of it.

* it is not tied to one architecture - any :class:`~orca.training.models.base_model.OrcaModel`
  accepts one, and sizes its first layer from :attr:`OrcaModel.expanded_dim`,
* it is tuned like any other hyperparameter, because its search space merges
  into the study alongside the model's,
* it is exported for free, since it is part of the graph ``torch.onnx.export``
  traces - no separate preprocessing has to be reproduced at inference time.

Basis expansions see **normalized** inputs, in ``(batch, n_inputs)`` layout: the dataset
normalizes before the model runs, and :class:`~orca.training.onnx_wrapper.ONNXWrapper`
does the same inside the exported graph.

Why expand frequency at all: ORCA's per-point models take frequency as a single
raw column among the geometry parameters, so the whole frequency response has to
be learned through one scalar. Giving the network a basis expansion of that
column on top allows the network to more easily capture complex frequency dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import optuna
import torch
import torch.nn as nn

from orca.training.spec import IOSpec


class BasisExpansion(nn.Module, ABC):
    """Maps a model input tensor to a wider one, before the first layer.

    Subclasses must implement :meth:`forward` and :meth:`expanded_dim`, and are
    built through :meth:`from_spec` so the model never has to know which basis
    it was given.
    """

    @abstractmethod
    def expanded_dim(self, input_dim: int) -> int:
        """Width of the tensor :meth:`forward` returns, given an input of ``input_dim``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand a normalized input batch of shape ``(batch, input_dim)``."""

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: IOSpec, hyperparameters: dict[str, Any]) -> "BasisExpansion":
        """Build a basis expansion for ``spec``, configured by ``hyperparameters``.

        Implementations must tolerate extra keys: the trainer's and the model's
        hyperparameters arrive through the same dictionary.
        """

    @staticmethod
    def hyperparameter_search_space() -> dict[str, Any]:
        """Basis hyperparameters for optuna tuning. Empty when there is nothing to tune."""
        return {}


_BASIS_REGISTRY: dict[str, type[BasisExpansion]] = {}


def register_basis(name: str) -> Callable[[type[BasisExpansion]], type[BasisExpansion]]:
    """Class decorator registering a basis expansion under a short name."""

    def decorator(cls: type[BasisExpansion]) -> type[BasisExpansion]:
        if name in _BASIS_REGISTRY and _BASIS_REGISTRY[name] is not cls:
            raise ValueError(f"Basis expansion '{name}' is already registered.")
        _BASIS_REGISTRY[name] = cls
        return cls

    return decorator


def get_basis_class(basis: str | type[BasisExpansion]) -> type[BasisExpansion]:
    """Resolve a registered basis name (or pass a BasisExpansion class straight through)."""
    if isinstance(basis, type) and issubclass(basis, BasisExpansion):
        return basis
    if isinstance(basis, str):
        try:
            return _BASIS_REGISTRY[basis]
        except KeyError:
            raise ValueError(
                f"Unknown basis '{basis}'. Available basis expansions: {available_bases()}"
            ) from None
    raise TypeError(f"Expected a BasisExpansion subclass or a registered name, got {basis!r}.")


def available_bases() -> list[str]:
    """Names of all registered basis expansions."""
    return sorted(_BASIS_REGISTRY)


@register_basis("identity")
class IdentityBasis(BasisExpansion):
    """Passes the input through unchanged. The default, so expansion is opt-in."""

    def expanded_dim(self, input_dim: int) -> int:
        return input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def from_spec(cls, spec: IOSpec, hyperparameters: dict[str, Any]) -> "IdentityBasis":
        return cls()


@register_basis("chebyshev")
class ChebyshevBasis(BasisExpansion):
    """Appends Chebyshev polynomials of one input column, usually frequency.

    The original columns are passed through untouched and ``degree`` extra
    columns are appended, holding T1..T_degree of the selected column mapped from
    ``domain`` onto [-1, 1].

    The mapped value is clamped to [-1, 1] before the recurrence. Chebyshev
    polynomials grow like ``cosh(n * arccosh(x))`` outside that interval, so a
    frequency slightly beyond the range seen during training would otherwise
    produce values orders of magnitude larger than anything the model was fitted
    on. Clamping saturates the basis instead; the un-clamped column is still
    passed through, so no information is lost.

    Args:
        column (int): Index of the input column to expand.
        degree (int): Highest polynomial order. Also the number of columns added.
        domain (tuple[float, float]): Range of the selected column in the model's
            *normalized* input space. The default matches a min-max normalizer,
            which maps every input onto [0, 1].
    """

    def __init__(
        self,
        column: int,
        degree: int = 8,
        domain: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        if degree < 1:
            raise ValueError(f"ChebyshevBasis needs degree >= 1, got {degree}.")
        low, high = domain
        if high <= low:
            raise ValueError(f"ChebyshevBasis needs domain low < high, got {domain}.")

        self.column = column
        self.degree = degree
        self.domain = domain

    def expanded_dim(self, input_dim: int) -> int:
        return input_dim + self.degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.domain
        t1 = (2.0 * (x[:, self.column] - low) / (high - low) - 1.0).clamp(-1.0, 1.0)

        # Chebyshev recurrence: T_n = 2 * t * T_(n-1) - T_(n-2), from T_0 = 1, T_1 = t.
        terms = [torch.ones_like(t1), t1]
        for _ in range(2, self.degree + 1):
            terms.append(2.0 * t1 * terms[-1] - terms[-2])

        # T_0 is the constant 1 and would only duplicate the bias of the next layer.
        return torch.cat([x, torch.stack(terms[1:], dim=1)], dim=1)

    @classmethod
    def from_spec(cls, spec: IOSpec, hyperparameters: dict[str, Any]) -> "ChebyshevBasis":
        column = hyperparameters.get("basis_column")
        if column is None:
            if "frequency" not in spec.input_names:
                raise ValueError(
                    "ChebyshevBasis defaults to expanding the 'frequency' column, but "
                    f"the dataset's inputs are {list(spec.input_names)}. Pass an explicit "
                    "'basis_column' index in the hyperparameters."
                )
            column = spec.input_names.index("frequency")
        return cls(
            column=column,
            degree=hyperparameters.get("basis_degree", 8),
            domain=hyperparameters.get("basis_domain", (0.0, 1.0)),
        )

    @staticmethod
    def hyperparameter_search_space() -> dict[str, Any]:
        return {"basis_degree": optuna.distributions.IntDistribution(2, 16, step=2)}
