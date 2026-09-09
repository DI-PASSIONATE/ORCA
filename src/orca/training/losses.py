"""Loss functions for S-parameter regression.

Losses are ``nn.Module`` subclasses so they can be configured, moved between
devices and returned from :meth:`~orca.training.models.base_model.OrcaModel.default_loss`
like any other torch loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        log_cosh = torch.mean(torch.log(torch.cosh(pred - target)))
        return self.log_cosh_weight * log_cosh + self.mse_weight * self.mse(pred, target)
