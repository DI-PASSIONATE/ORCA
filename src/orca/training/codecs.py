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
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import skrf as rf

from orca.training.guarantees import PhysicsGuarantees

if TYPE_CHECKING:
    # Only for annotations: `import orca` loads this module, and has to work in a
    # simulation-only install without PyTorch.
    import torch


class OutputCodec(ABC):
    """Bidirectional mapping between a network response and a model target.

    The contract is one row per frequency point: :meth:`encode` returns shape
    ``(n_freq, output_dim)`` and :meth:`decode` accepts the same. Per-point
    datasets emit one sample per row, band datasets keep the rows together.

    Args:
        n_ports (int): Number of ports of the networks handled by this codec.
    """

    #: Physical properties the representation enforces by construction. A codec
    #: that cannot express a violation guarantees the property regardless of the
    #: architecture sitting behind it.
    guarantees: ClassVar[PhysicsGuarantees] = PhysicsGuarantees()

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

    def decode_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        """Differentiable :meth:`decode` of a batch of denormalized outputs, in torch.

        The loss terms that work on the S-matrix (``ModelTrainer(admittance_weight=...,
        passivity_weight=...)``) need it; a codec that does not implement it cannot be
        trained with them.

        Args:
            raw (torch.Tensor): Outputs of shape ``(batch, output_dim)``.

        Returns:
            torch.Tensor: Complex S-matrices of shape ``(batch, n_ports, n_ports)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement decode_tensor, which the admittance "
            "and passivity loss terms need."
        )

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

    def decode_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        import torch  # Local: see the TYPE_CHECKING import

        pairs = raw.reshape(-1, self.n_ports, self.n_ports, 2)
        return torch.complex(pairs[..., 0], pairs[..., 1])


class UpperTriangleReImCodec(OutputCodec):
    """The upper triangle of a reciprocal S-matrix, as interleaved real/imag pairs.

    A passive reciprocal structure satisfies ``S = S.T``, so the lower triangle
    carries no information: for 6 ports there are 21 unique complex entries, not
    36. Predicting only ``S_ij`` with ``i <= j`` and mirroring on decode roughly
    halves the output dimension and makes reciprocity structural — the model
    cannot emit an asymmetric S-matrix, so a downstream optimizer cannot exploit
    one.

    Produces ``n_ports * (n_ports + 1)`` values per frequency point, ordered
    ``S11_real, S11_imag, S12_real, ..., S1N_imag, S22_real, ...`` (row-major over
    the upper triangle, diagonal included).

    Simulated data is never *exactly* symmetric, so :meth:`encode` averages each
    entry with its transpose rather than discarding the lower triangle: that is
    the projection onto the reciprocal subspace, and it uses both numbers instead
    of throwing one away.

    Note:
        Only valid for reciprocal structures. Anything with a non-reciprocal
        element (ferrite, active device) needs :class:`FlatReImCodec`.
    """

    guarantees: ClassVar[PhysicsGuarantees] = PhysicsGuarantees(reciprocal=True)

    @property
    def n_entries(self) -> int:
        """Number of unique complex entries, i.e. the size of the upper triangle."""
        return self.n_ports * (self.n_ports + 1) // 2

    @property
    def output_dim(self) -> int:
        return 2 * self.n_entries

    @property
    def output_names(self) -> list[str]:
        return [
            f"S{i + 1}{j + 1}_{part}"
            for i in range(self.n_ports)
            for j in range(i, self.n_ports)
            for part in ("real", "imag")
        ]

    def _triangle_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Row and column indices of the upper triangle, in output order."""
        return np.triu_indices(self.n_ports)

    def encode(self, ntwk: rf.Network) -> np.ndarray:
        s = ntwk.s  # (n_freq, n_ports, n_ports)
        symmetric = 0.5 * (s + np.swapaxes(s, -1, -2))
        rows, cols = self._triangle_indices()
        entries = symmetric[:, rows, cols]  # (n_freq, n_entries)
        interleaved = np.stack((entries.real, entries.imag), axis=-1)
        return interleaved.reshape(s.shape[0], -1).astype(np.float32)

    def decode(self, raw: np.ndarray) -> np.ndarray:
        n = self.n_ports
        raw = np.asarray(raw).reshape(-1, self.n_entries, 2)
        entries = raw[..., 0] + 1j * raw[..., 1]

        rows, cols = self._triangle_indices()
        s = np.zeros((raw.shape[0], n, n), dtype=np.complex64)
        s[:, rows, cols] = entries
        s[:, cols, rows] = entries  # mirror; the diagonal is written twice, harmlessly
        return s

    def decode_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        import torch  # Local: see the TYPE_CHECKING import

        n = self.n_ports
        pairs = raw.reshape(-1, self.n_entries, 2)
        entries = torch.complex(pairs[..., 0], pairs[..., 1])
        # Gathered rather than written into a zero matrix, so autograd sees one plain
        # indexing operation: entry (i, j) and entry (j, i) both read triangle entry k.
        rows, cols = self._triangle_indices()
        entry_of = np.empty((n, n), dtype=np.int64)
        entry_of[rows, cols] = entry_of[cols, rows] = np.arange(self.n_entries)
        index = torch.as_tensor(entry_of.reshape(-1), device=raw.device)
        return entries[:, index].reshape(-1, n, n)
