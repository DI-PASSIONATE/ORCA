"""Output representations for S-parameter models.

An :class:`OutputCodec` owns the translation between a full-wave result
(``skrf.Network``) and the flat tensor a model actually regresses on:

* datasets call :meth:`encode` to build training targets,
* inference calls :meth:`decode` / :meth:`to_network` to get an S-matrix back,
* the ONNX exporter uses :attr:`output_names` for the model signature.

Swapping the codec is how a new output representation (upper-triangle only,
Y-parameters, pole-residue, ...) enters the pipeline without touching the
dataset, the trainer or the export stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import skrf as rf


class OutputCodec(ABC):
    """Bidirectional mapping between a network response and a model target.

    The contract is one row per frequency point: :meth:`encode` returns shape
    ``(n_freq, output_dim)`` and :meth:`decode` accepts the same. Per-point
    datasets emit one sample per row, band datasets keep the rows together.

    Args:
        n_ports (int): Number of ports of the networks handled by this codec.
    """

    def __init__(self, n_ports: int):
        self.n_ports = n_ports

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of real values produced per frequency point."""

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Names of the output values, in the order :meth:`encode` produces them."""

    @abstractmethod
    def encode(self, ntwk: rf.Network) -> np.ndarray:
        """Convert a network into training targets of shape (n_freq, output_dim)."""

    @abstractmethod
    def decode(self, raw: np.ndarray) -> np.ndarray:
        """Convert model outputs (n_freq, output_dim) into a complex S-matrix.

        Returns:
            np.ndarray: Complex array of shape (n_freq, n_ports, n_ports).
        """

    def to_network(self, raw: np.ndarray, frequencies: np.ndarray) -> rf.Network:
        """Decode model outputs and wrap them in a ``skrf.Network``."""
        s = self.decode(np.asarray(raw))
        return rf.Network(frequency=frequencies, s=s, f_unit="Hz")


class FlatReImCodec(OutputCodec):
    """Every S-parameter as an interleaved real/imaginary pair.

    Produces ``2 * n_ports**2`` values per frequency point, ordered
    ``S11_real, S11_imag, S12_real, ...``. This is ORCA's historical output
    layout and the one COBRA expects from exported ONNX models.
    """

    @property
    def output_dim(self) -> int:
        return 2 * self.n_ports**2

    @property
    def output_names(self) -> list[str]:
        return [
            f"S{i + 1}{j + 1}_{part}"
            for i in range(self.n_ports)
            for j in range(self.n_ports)
            for part in ("real", "imag")
        ]

    def encode(self, ntwk: rf.Network) -> np.ndarray:
        s = ntwk.s  # (n_freq, n_ports, n_ports)
        return np.stack((s.real, s.imag), axis=-1).reshape(s.shape[0], -1).astype(np.float32)

    def decode(self, raw: np.ndarray) -> np.ndarray:
        n = self.n_ports
        raw = np.asarray(raw).reshape(-1, n, n, 2)
        return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)
