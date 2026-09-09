"""Training loop for ORCA surrogate models.

:class:`Trainer` owns everything about *how* a model is fitted - optimizer,
schedule, early stopping, progress reporting - while the model owns *what* is
fitted (architecture and loss). :class:`TrainingConfig` separates the
hyperparameters the trainer owns from the architecture hyperparameters a model
declares, so the two can be tuned together but configured independently.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import optuna
import torch
import torch.nn as nn
import tqdm
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

from orca.training.models.base_model import OrcaModel


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    """Hyperparameters owned by the trainer rather than by any architecture.

    Attributes:
        epochs: Maximum number of epochs to run.
        batch_size: Mini-batch size for both training and validation.
        learning_rate: Initial learning rate handed to the optimizer.
        patience: Epochs without validation improvement before stopping early.
        optimizer_cls: Optimizer class, constructed as ``optimizer_cls(params, lr=...)``.
        scheduler_factor: Factor by which ReduceLROnPlateau scales the learning rate.
        scheduler_patience: Plateau length, in epochs, before the scheduler reacts.
        device: Device to train on.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    patience: int = 10
    optimizer_cls: type[Optimizer] = AdamW
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    device: torch.device = field(default_factory=default_device)

    @classmethod
    def from_hyperparameters(cls, hyperparameters: dict[str, Any], **overrides) -> "TrainingConfig":
        """Build a config from a hyperparameter dict, ignoring architecture keys.

        Args:
            hyperparameters (dict[str, Any]): Combined trainer and model hyperparameters.
            **overrides: Explicit values that take precedence over the dictionary.
        """
        known = {"epochs", "batch_size", "learning_rate", "patience"}
        values = {k: v for k, v in hyperparameters.items() if k in known}
        values.update(overrides)
        return cls(**values)

    @staticmethod
    def search_space() -> dict[str, Any]:
        """Optuna search space for the trainer's own hyperparameters."""
        return {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "batch_size": [32, 64, 128, 256, 512],
            "epochs": optuna.distributions.IntDistribution(5, 50, step=5),
        }


@dataclass
class EpochResult:
    """Losses recorded for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float


@dataclass
class TrainingResult:
    """Outcome of a :meth:`Trainer.fit` call.

    Attributes:
        model: The trained model, restored to its best validation checkpoint.
        best_loss: Lowest validation loss observed.
        history: Per-epoch losses, in order.
        stopped_early: Whether early stopping ended the run before ``epochs``.
    """

    model: nn.Module
    best_loss: float
    history: list[EpochResult]
    stopped_early: bool

    @property
    def epochs_run(self) -> int:
        return len(self.history)


class Trainer:
    """Fits a model to a dataset with early stopping and LR scheduling.

    Args:
        config (TrainingConfig | None): Training hyperparameters. Defaults are used if omitted.
        criterion (Callable | None): Loss to optimize. If omitted, the model's
            :meth:`~orca.training.models.base_model.OrcaModel.default_loss` is used.
        progress_callback (Callable | None): Called as
            ``(stage_name, current_epoch, total_epochs, message)`` after every epoch.
        stage_name (str): Label passed to ``progress_callback``.
        verbose (bool): Whether to print per-epoch losses and show progress bars.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        stage_name: str = "Training",
        verbose: bool = True,
    ):
        self.config = config or TrainingConfig()
        self.criterion = criterion
        self.progress_callback = progress_callback
        self.stage_name = stage_name
        self.verbose = verbose

    def resolve_criterion(self, model: nn.Module) -> Callable:
        """The configured loss, or the one the model asks to be trained with."""
        if self.criterion is not None:
            return self.criterion
        if isinstance(model, OrcaModel):
            return model.default_loss()
        return nn.L1Loss()

    def fit(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainingResult:
        """Train ``model``, keeping the weights with the lowest validation loss.

        Args:
            model (nn.Module): Model to train. Moved to the configured device.
            train_dataset (Dataset): Samples to optimize on.
            val_dataset (Dataset): Samples used for early stopping and scheduling.

        Returns:
            TrainingResult: The best model, its loss, and the per-epoch history.
        """
        config = self.config
        model.to(config.device)
        criterion = self.resolve_criterion(model)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        optimizer = config.optimizer_cls(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.scheduler_factor, patience=config.scheduler_patience
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: list[EpochResult] = []
        stopped_early = False

        self._report(0, "Starting training")

        for epoch in range(config.epochs):
            train_loss = self._train_epoch(model, criterion, optimizer, train_loader)
            val_loss = self._validate(model, criterion, val_loader)
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            history.append(
                EpochResult(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
            )

            message = f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            self._report(epoch + 1, message)
            if self.verbose:
                print(f"Epoch {epoch + 1:4d} | {message}")

            if patience_counter >= config.patience:
                stopped_early = True
                if self.verbose:
                    print("Early stopping triggered")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return TrainingResult(
            model=model, best_loss=best_loss, history=history, stopped_early=stopped_early
        )

    def evaluate(
        self,
        model: nn.Module,
        dataset: Dataset,
        criterion: Optional[Callable] = None,
        batch_size: int | None = None,
    ) -> float:
        """Mean loss of ``model`` over ``dataset``, without updating any weights."""
        model.to(self.config.device)
        criterion = criterion or self.resolve_criterion(model)
        loader = DataLoader(dataset, batch_size=batch_size or self.config.batch_size)
        return self._run_eval(model, criterion, loader, desc="Testing")

    def _train_epoch(self, model, criterion, optimizer, loader) -> float:
        model.train()
        total = 0.0

        for x, y in tqdm.tqdm(loader, desc="Training", leave=False, disable=not self.verbose):
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            total += loss.item()

        return total / len(loader)

    def _validate(self, model, criterion, loader) -> float:
        return self._run_eval(model, criterion, loader, desc="Validation", show_progress=False)

    def _run_eval(self, model, criterion, loader, desc: str, show_progress: bool = True) -> float:
        model.eval()
        total = 0.0

        iterator = loader
        if show_progress:
            iterator = tqdm.tqdm(loader, desc=desc, leave=False, disable=not self.verbose)

        with torch.no_grad():
            for x, y in iterator:
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                total += criterion(model(x), y).item()

        return total / len(loader)

    def _report(self, epoch: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(self.stage_name, epoch, self.config.epochs, message)
