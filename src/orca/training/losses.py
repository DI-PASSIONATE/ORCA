"""Loss functions for S-parameter regression.

Losses are ``nn.Module`` subclasses so they can be configured, moved between
devices and returned from :meth:`~orca.training.models.base_model.OrcaModel.default_loss`
like any other torch loss.

:class:`SParameterLoss` adds optional terms that judge the prediction as a circuit
rather than as a vector of numbers (the relative error of its admittance matrix, a
passivity penalty) and per-sample weights, such as those of :func:`srf_sample_weights`,
which weight the frequency points above each geometry's self-resonance differently.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from orca.logger import logger
from orca.training.spec import FrequencyMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from orca.training.codecs import OutputCodec
    from orca.training.datasets.base_dataset import BaseDataset
    from orca.training.normalize import Normalizer

#: Conductance floor of :func:`admittance_error`, relative to the magnitude of the
#: admittance matrix: keeps the conductance term finite where the conductance vanishes.
CONDUCTANCE_FLOOR = 1e-2


class ComplexMSELoss(nn.Module):
    """Mean squared error over interleaved real/imaginary output pairs.

    Equivalent to a plain MSE over the flat vector, but reshapes to (batch, N, 2)
    first so the pairing is explicit.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = pred.shape[1] // 2
        pred = pred.view(-1, n, 2)
        target = target.view(-1, n, 2)
        return torch.mean((pred - target) ** 2)


class MSEPlusLogCoshLoss(nn.Module):
    """Weighted sum of a log-cosh term and a complex MSE term.

    log-cosh behaves like L1 for large errors and like L2 for small ones, which
    keeps outliers from dominating while staying smooth near the optimum.

    Args:
        log_cosh_weight (float): Weight applied to the log-cosh term.
        mse_weight (float): Weight applied to the complex MSE term.
    """

    def __init__(self, log_cosh_weight: float = 2.0, mse_weight: float = 1.0):
        super().__init__()
        self.log_cosh_weight = log_cosh_weight
        self.mse_weight = mse_weight
        self.mse = ComplexMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        abs_error = torch.abs(error)
        log_cosh = torch.mean(
            abs_error + torch.nn.functional.softplus(-2.0 * abs_error) - math.log(2.0)
        )
        return self.log_cosh_weight * log_cosh + self.mse_weight * self.mse(pred, target)


def s_to_y(s: torch.Tensor) -> torch.Tensor:
    """Admittance matrices of a batch of S-matrices, in units of the reference admittance.

    ``y = (I + S)^-1 (I - S)``, which is ``Z0 * Y`` for a common, real reference
    impedance ``Z0``. The two factors commute, so the inverse is applied by a solve.

    Args:
        s (torch.Tensor): Complex S-matrices of shape ``(batch, n_ports, n_ports)``.

    Returns:
        torch.Tensor: Normalized admittance matrices of the same shape.
    """
    eye = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return torch.linalg.solve(eye + s, eye - s)


def admittance_error(s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """Relative error of the predicted admittance matrix, per sample.

    A small error in S can be a large error in the inductance and quality factor: at
    low frequency an inductor is close to a short between its ports, where Y = 1/(R +
    jwL) reacts strongly to S. The sum of two relative L1 errors over all entries:

    * of Y as a whole, which follows the reactance (L, C, the self-resonance), and
    * of its real part, the conductance, which sets the losses and with them Q. It is
      a few percent of |Y| for a high-Q inductor, so the first term barely sees it.

    Both are scaled by the reference matrix of each sample, so every frequency point
    counts alike whatever the magnitude of its Y. The conductance is floored at
    :data:`CONDUCTANCE_FLOOR` of that magnitude.

    Args:
        s_pred (torch.Tensor): Predicted complex S-matrices, ``(batch, n, n)``.
        s_true (torch.Tensor): Reference complex S-matrices, same shape.

    Returns:
        torch.Tensor: The error of each sample, shape ``(batch,)``.
    """
    y_true = s_to_y(s_true)
    error = s_to_y(s_pred) - y_true

    def entry_sum(x: torch.Tensor) -> torch.Tensor:
        return x.abs().sum(dim=(-2, -1))

    scale = entry_sum(y_true) + 1e-12
    conductance = entry_sum(y_true.real) + CONDUCTANCE_FLOOR * scale
    return entry_sum(error) / scale + entry_sum(error.real) / conductance


def passivity_violation(s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """How far the largest singular value of each predicted S-matrix exceeds 1.

    A passive network has ``sigma_max(S) <= 1``. The reference may exceed that
    slightly, as the DC point extrapolated by ``touchstone_type="dc_deembedded"``
    does; a prediction is only penalised beyond ``max(1, sigma_max(S_true))``, so the
    penalty never pulls the model away from its own training data.

    Args:
        s_pred (torch.Tensor): Predicted complex S-matrices, ``(batch, n, n)``.
        s_true (torch.Tensor): Reference complex S-matrices, same shape.

    Returns:
        torch.Tensor: The violation of each sample, zero where passive; ``(batch,)``.
    """
    allowed = largest_singular_value(s_true).clamp(min=1.0)
    return torch.relu(largest_singular_value(s_pred) - allowed)


def largest_singular_value(s: torch.Tensor) -> torch.Tensor:
    """``sigma_max`` of each matrix of a batch, as the root of the top eigenvalue of S^H S.

    Equal to ``torch.linalg.matrix_norm(s, ord=2)``, but about five times faster for a
    batch of small matrices on a GPU, where batched SVDs are slow.
    """
    top = torch.linalg.eigvalsh(s.mH @ s)[..., -1]
    # Floored, so the square root has a finite gradient for an all-zero matrix
    return top.clamp(min=1e-12).sqrt()


class SParameterLoss(nn.Module):
    """A model's data loss, plus optional circuit-level terms and per-sample weights.

    Every term is computed per sample and summed with its weight; the per-sample sum
    is then averaged, weighted by ``sample_weights`` if given. Those weights should
    average 1 over a dataset (as :func:`srf_sample_weights` returns them), so the
    loss of a whole epoch is their weighted mean and stays on the scale of the data
    loss alone.

    Args:
        data_loss (Callable): Elementwise loss on the normalized outputs, usually
            the model's :meth:`~orca.training.models.base_model.OrcaModel.default_loss`.
            It needs a ``reduction`` attribute, as torch's own losses have; a copy
            set to ``"none"`` is used.
        codec (OutputCodec): Layout of the outputs; its
            :meth:`~orca.training.codecs.OutputCodec.decode_tensor` rebuilds the S-matrices.
        output_normalizer (Normalizer | None): Undoes the output normalization before
            the S-matrices are rebuilt.
        admittance_weight (float): Weight of :func:`admittance_error`; 0 leaves it out.
        passivity_weight (float): Weight of :func:`passivity_violation`; 0 leaves it out.
    """

    def __init__(
        self,
        data_loss: Callable[..., torch.Tensor],
        codec: OutputCodec,
        output_normalizer: Normalizer | None = None,
        admittance_weight: float = 0.0,
        passivity_weight: float = 0.0,
    ):
        super().__init__()
        if not hasattr(data_loss, "reduction"):
            raise TypeError(
                f"SParameterLoss needs an elementwise data loss with a 'reduction' attribute "
                f"(such as nn.L1Loss), got {type(data_loss).__name__}."
            )
        if admittance_weight < 0 or passivity_weight < 0:
            raise ValueError("Loss term weights must not be negative.")
        # Typed Any: the reduction attribute is torch's convention, not part of Callable
        elementwise: Any = copy.copy(data_loss)
        elementwise.reduction = "none"
        self.data_loss = elementwise
        self.codec = codec
        self.output_normalizer = output_normalizer
        self.admittance_weight = admittance_weight
        self.passivity_weight = passivity_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.data_loss(pred, target).reshape(len(pred), -1).mean(dim=1)
        if self.admittance_weight > 0 or self.passivity_weight > 0:
            s_pred, s_true = self._s_matrices(pred), self._s_matrices(target)
            if self.admittance_weight > 0:
                loss = loss + self.admittance_weight * admittance_error(s_pred, s_true)
            if self.passivity_weight > 0:
                loss = loss + self.passivity_weight * passivity_violation(s_pred, s_true)
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()

    def _s_matrices(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.output_normalizer is not None:
            outputs = self.output_normalizer.denormalize(outputs)
        return self.codec.decode_tensor(outputs)


def first_self_resonance(frequencies: np.ndarray, s: np.ndarray) -> float:
    """The first self-resonance frequency of a network, from its S-parameters.

    The frequency at which the first inductive mode of the network turns capacitive:
    the susceptance matrix ``Im(Y)`` (its Hermitian part) has one negative
    eigenvalue per inductive mode, and the resonance is where their number first
    drops, interpolated linearly. This needs no knowledge of the ports: for a
    center-tapped inductor it finds the differential self-resonance (within the
    frequency step, on the InductorOcta results), since the differential mode
    resonates first. The modes are counted at the lowest frequency above DC, where Y
    is still almost real.

    Args:
        frequencies (np.ndarray): Frequencies in Hz, shape ``(n_freq,)``, any order.
        s (np.ndarray): Complex S-matrices at those frequencies, ``(n_freq, n, n)``.

    Returns:
        float: The resonance in Hz; ``inf`` if there is none in the band, or if the
        network has no inductive mode at its lowest frequency.
    """
    order = np.argsort(frequencies)
    frequencies, s = frequencies[order], s[order]
    eye = np.eye(s.shape[-1])
    try:
        y = np.linalg.solve(eye + s, eye - s)
    except np.linalg.LinAlgError:
        logger.warning("Cannot convert an S-matrix to Y (a short circuit); no resonance found.")
        return math.inf
    susceptance = (y - np.conj(np.swapaxes(y, -1, -2))) / 2j
    eigenvalues = np.linalg.eigvalsh(susceptance)  # ascending, per frequency
    # Relative to the largest mode, so rounding noise around zero is not a mode
    threshold = -1e-6 * np.abs(eigenvalues).max(axis=1)
    n_inductive = (eigenvalues < threshold[:, None]).sum(axis=1)

    ac = np.flatnonzero(frequencies > 0)
    if ac.size < 2 or n_inductive[ac[0]] == 0:
        return math.inf
    reference = n_inductive[ac[0]]
    dropped = ac[n_inductive[ac] < reference]
    if dropped.size == 0:
        return math.inf

    k = dropped[0]
    # The eigenvalue that turned is the reference-th smallest one
    before, after = eigenvalues[k - 1, reference - 1], eigenvalues[k, reference - 1]
    f0, f1 = frequencies[k - 1], frequencies[k]
    if after <= before:
        return float(f1)
    return float(np.clip(f0 - before * (f1 - f0) / (after - before), f0, f1))


def srf_sample_weights(dataset: BaseDataset, above_srf_weight: float) -> torch.Tensor:
    """Per-sample loss weights that weight frequencies above the self-resonance apart.

    Below its first self-resonance (:func:`first_self_resonance`) a passive is used
    as what it is, an inductor say; above it the response swings through resonances
    that the network has to spend capacity on, but that a designer rarely needs.
    Each frequency point of a geometry above that geometry's resonance is weighted
    ``above_srf_weight`` relative to the points below it. Geometries that do not
    resonate in the band keep weight 1 everywhere.

    Args:
        dataset (BaseDataset): A loaded per-point dataset with a ``frequency`` input.
        above_srf_weight (float): Weight of the points above the resonance, relative
            to those below; positive.

    Returns:
        torch.Tensor: One weight per sample, on the dataset's device, averaging 1.
    """
    if type(dataset).frequency_mode is not FrequencyMode.PER_POINT:
        raise ValueError(
            f"Weighting by self-resonance needs a per-point dataset, but "
            f"{type(dataset).__name__} lays out frequency as {type(dataset).frequency_mode.name}."
        )
    if "frequency" not in dataset.input_param_names:
        raise ValueError(
            f"Weighting by self-resonance needs a 'frequency' input, but the inputs are "
            f"{dataset.input_param_names}."
        )
    if above_srf_weight <= 0:
        raise ValueError(f"above_srf_weight must be positive, got {above_srf_weight}.")

    inputs, targets = dataset.inputs, dataset.targets
    if dataset.input_normalizer is not None:
        inputs = dataset.input_normalizer.denormalize(inputs)
    if dataset.output_normalizer is not None:
        targets = dataset.output_normalizer.denormalize(targets)
    column = dataset.input_param_names.index("frequency")
    frequencies = inputs[:, column].double().cpu().numpy()
    s = dataset.codec.decode(targets.cpu().numpy())

    # The samples of each geometry, found by one sort rather than a scan per geometry
    _, group_of, counts = np.unique(
        np.asarray(dataset.sample_groups), return_inverse=True, return_counts=True
    )
    order = np.argsort(group_of, kind="stable")
    resonance = np.array(
        [
            first_self_resonance(frequencies[members], s[members])
            for members in np.split(order, np.cumsum(counts)[:-1])
        ]
    )

    above = frequencies > resonance[group_of]
    weights = np.where(above, above_srf_weight, 1.0)
    weights /= weights.mean()
    in_band = resonance[np.isfinite(resonance)]
    logger.info(
        f"Self-resonance found in the band for {in_band.size} of {resonance.size} geometries"
        + (f" (median {np.median(in_band) / 1e9:.3g} GHz)" if in_band.size else "")
        + f"; {above.mean():.1%} of the frequency points lie above it and are weighted "
        f"{above_srf_weight:g} relative to the rest."
    )
    return torch.as_tensor(weights, dtype=torch.float32, device=dataset.inputs.device)
