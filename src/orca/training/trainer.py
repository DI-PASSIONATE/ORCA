"""Training loop for ORCA surrogate models.

:class:`Trainer` owns everything about *how* a model is fitted - optimizer,
schedule, early stopping, progress reporting - while the model owns *what* is
fitted (architecture and loss). :class:`TrainingConfig` separates the
hyperparameters the trainer owns from the architecture hyperparameters a model
declares, so the two can be tuned together but configured independently.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optuna
import torch
import tqdm
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from orca.training.models.base_model import OrcaModel


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#: Batch sizes the tuner chooses from when none are given.
DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512)

#: Learning-rate schedules the trainer supports, see :attr:`TrainingConfig.lr_schedule`.
LR_SCHEDULES = ("cosine", "plateau")


@dataclass
class TrainingConfig:
    """Hyperparameters owned by the trainer rather than by any architecture.

    Attributes:
        epochs: Maximum number of epochs to run.
        batch_size: Mini-batch size for both training and validation.
        learning_rate: Initial learning rate handed to the optimizer.
        weight_decay: Weight decay handed to the optimizer, as a regularization. The
            default is AdamW's own.
        patience: Epochs without validation improvement before stopping early.
        optimizer_cls: Optimizer factory, called as
            ``optimizer_cls(params, lr=..., weight_decay=...)``.
        lr_schedule: How the learning rate changes after the warmup. ``"cosine"`` decays
            it every step along a half cosine, reaching ``min_lr_ratio`` times the
            initial rate after ``epochs`` epochs. ``"plateau"`` lowers it by
            ``scheduler_factor`` whenever the validation loss stalls.
        warmup_epochs: Epochs over which the learning rate rises linearly from almost
            zero to ``learning_rate``, step by step. Fractions are allowed; 0 disables it.
            Without a warmup, AdamW's first steps on freshly initialised weights can
            throw a large network far off, and a run then spends epochs recovering.
        min_lr_ratio: Final learning rate of the cosine schedule, relative to
            ``learning_rate``.
        grad_clip_norm: Largest total gradient norm of an optimizer step; larger
            gradients are scaled down to it. Keeps a single bad batch from undoing the
            training so far. ``None`` disables clipping.
        scheduler_factor: Factor by which the plateau schedule scales the learning rate.
        scheduler_patience: Plateau length, in epochs, before the plateau schedule
            reacts. Keep it below ``patience``, or early stopping ends the run before
            the learning rate is ever lowered.
        allow_tf32: Let matrix multiplies round their inputs to TF32 (10 mantissa bits)
            while training, on GPUs with TF32 tensor cores (Ampere and newer).
        device: Device to train on.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 10
    optimizer_cls: Callable[..., Optimizer] = AdamW
    lr_schedule: str = "cosine"
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 1e-2
    grad_clip_norm: float | None = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    allow_tf32: bool = False
    device: torch.device = field(default_factory=default_device)

    def __post_init__(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"Unknown lr_schedule {self.lr_schedule!r}; choose one of {list(LR_SCHEDULES)}."
            )
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must not be negative, got {self.warmup_epochs}.")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must lie in [0, 1], got {self.min_lr_ratio}.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {self.grad_clip_norm}."
            )

    @classmethod
    def from_hyperparameters(cls, hyperparameters: dict[str, Any], **overrides) -> TrainingConfig:
        """Build a config from a hyperparameter dict, ignoring architecture keys.

        Args:
            hyperparameters (dict[str, Any]): Combined trainer and model hyperparameters.
            **overrides: Explicit values that take precedence over the dictionary.
        """
        known = {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "patience",
            "lr_schedule",
            "warmup_epochs",
            "min_lr_ratio",
            "grad_clip_norm",
            "allow_tf32",
        }
        values = {k: v for k, v in hyperparameters.items() if k in known}
        values.update(overrides)
        return cls(**values)

    @staticmethod
    def search_space(
        batch_sizes: Sequence[int] | None = None, regularization: bool = False
    ) -> dict[str, Any]:
        """Optuna search space for the trainer's own hyperparameters.

        Args:
            batch_sizes (Sequence[int] | None): Batch sizes to choose from. ``None``
                uses :data:`DEFAULT_BATCH_SIZES`.
            regularization (bool): Also search the weight decay. Otherwise every trial
                uses the default.
        """
        space: dict[str, Any] = {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "batch_size": list(batch_sizes or DEFAULT_BATCH_SIZES),
        }
        if regularization:
            space["weight_decay"] = optuna.distributions.FloatDistribution(1e-6, 1e-1, log=True)
        return space


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

    model: OrcaModel
    best_loss: float
    history: list[EpochResult]
    stopped_early: bool

    @property
    def epochs_run(self) -> int:
        return len(self.history)


def _stacked_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The inputs and targets of ``dataset`` as two tensors, if it keeps them stacked.

    Datasets exposing ``tensors`` (ORCA's datasets and ``TensorDataset``) and ``Subset``
    views of them qualify; anything else returns ``None``.
    """
    if isinstance(dataset, Subset):
        stacked = _stacked_tensors(dataset.dataset)
        if stacked is None:
            return None
        indices = torch.as_tensor(dataset.indices, dtype=torch.long, device=stacked[0].device)
        return stacked[0][indices], stacked[1][indices]
    tensors = getattr(dataset, "tensors", None)
    if isinstance(tensors, tuple) and len(tensors) == 2:  # (inputs, targets)
        return tensors
    return None


class _TensorBatches:
    """Mini-batches cut from two stacked tensors."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool):
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.inputs) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.inputs)
        if self.shuffle:
            order = torch.randperm(n, device=self.inputs.device)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                yield self.inputs[batch], self.targets[batch]
        else:
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                yield self.inputs[start:end], self.targets[start:end]


@contextlib.contextmanager
def matmul_precision(allow_tf32: bool) -> Iterator[None]:
    """Run the block with TF32 matrix multiplies allowed or not, then restore the setting."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def make_batches(
    dataset: Dataset, batch_size: int, shuffle: bool
) -> _TensorBatches | DataLoader:
    """Mini-batches of ``dataset``, sliced from stacked tensors where it has them."""
    stacked = _stacked_tensors(dataset)
    if stacked is not None:
        return _TensorBatches(*stacked, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class LearningRateSchedule:
    """The learning rate of a :class:`Trainer` run: a linear warmup, then the configured schedule.

    The warmup and the cosine decay set the rate before every optimizer step, so they
    are smooth even when one epoch is thousands of steps; the plateau schedule reacts
    to the validation loss once per epoch, starting after the warmup.

    Args:
        optimizer (Optimizer): Optimizer whose learning rate is controlled.
        config (TrainingConfig): Supplies the schedule and its parameters.
        steps_per_epoch (int): Optimizer steps in one epoch.
    """

    def __init__(self, optimizer: Optimizer, config: TrainingConfig, steps_per_epoch: int):
        self.optimizer = optimizer
        self.config = config
        self.warmup_steps = round(config.warmup_epochs * steps_per_epoch)
        self.total_steps = config.epochs * steps_per_epoch
        self.steps_taken = 0
        self.plateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=config.scheduler_factor, patience=config.scheduler_patience
            )
            if config.lr_schedule == "plateau"
            else None
        )

    def before_step(self) -> None:
        """Set the learning rate of the optimizer step about to be taken."""
        base = self.config.learning_rate
        step = self.steps_taken
        self.steps_taken += 1
        if step < self.warmup_steps:
            self._set((step + 1) / self.warmup_steps * base)
        elif self.plateau is None:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            minimum = self.config.min_lr_ratio * base
            self._set(minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * progress)))

    def after_epoch(self, val_loss: float) -> None:
        """Let the plateau schedule react to the epoch's validation loss."""
        if self.plateau is not None and self.steps_taken >= self.warmup_steps:
            self.plateau.step(val_loss)

    def _set(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate


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
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        stage_name: str = "Training",
        verbose: bool = True,
    ):
        self.config = config or TrainingConfig()
        self.criterion = criterion
        self.progress_callback = progress_callback
        self.stage_name = stage_name
        self.verbose = verbose

    def resolve_criterion(self, model: OrcaModel) -> Callable:
        """The configured loss, or the one the model asks to be trained with."""
        if self.criterion is not None:
            return self.criterion
        return model.default_loss()

    def fit(
        self,
        model: OrcaModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epoch_callback: Callable[[EpochResult], None] | None = None,
    ) -> TrainingResult:
        """Train ``model``, keeping the weights with the lowest validation loss.

        Args:
            model (OrcaModel): Model to train. Moved to the configured device.
            train_dataset (Dataset): Samples to optimize on.
            val_dataset (Dataset): Samples used for early stopping and scheduling.
            epoch_callback (Callable | None): Called with each epoch's result. An
                exception it raises ends training and propagates; the tuner prunes
                trials this way.

        Returns:
            TrainingResult: The best model, its loss, and the per-epoch history.
        """
        config = self.config
        model.to(config.device)
        criterion = self.resolve_criterion(model)

        train_loader = make_batches(train_dataset, config.batch_size, shuffle=True)
        val_loader = make_batches(val_dataset, config.batch_size, shuffle=False)

        optimizer = config.optimizer_cls(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        schedule = LearningRateSchedule(optimizer, config, steps_per_epoch=len(train_loader))

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        history: list[EpochResult] = []
        stopped_early = False

        self._report(0, "Starting training")

        with matmul_precision(config.allow_tf32):
            for epoch in range(config.epochs):
                train_loss = self._train_epoch(model, criterion, optimizer, schedule, train_loader)
                val_loss = self._validate(model, criterion, val_loader)
                schedule.after_epoch(val_loss)

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

                if epoch_callback is not None:
                    epoch_callback(history[-1])

                message = f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
                self._report(epoch + 1, message)
                if self.verbose:
                    # tqdm.write keeps the line clear of the epoch progress bar
                    tqdm.tqdm.write(f"Epoch {epoch + 1:4d} | {message}")

                if patience_counter >= config.patience:
                    stopped_early = True
                    if self.verbose:
                        tqdm.tqdm.write("Early stopping triggered")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return TrainingResult(
            model=model, best_loss=best_loss, history=history, stopped_early=stopped_early
        )

    def evaluate(
        self,
        model: OrcaModel,
        dataset: Dataset,
        criterion: Callable | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Mean loss of ``model`` over ``dataset``, without updating any weights."""
        model.to(self.config.device)
        criterion = criterion or self.resolve_criterion(model)
        loader = make_batches(dataset, batch_size or self.config.batch_size, shuffle=False)
        return self._run_eval(model, criterion, loader, desc="Testing")

    def _train_epoch(self, model, criterion, optimizer, schedule, loader) -> float:
        model.train()
        # Summed on the device and read once per epoch: reading it every step would make
        # the CPU wait for the GPU after each batch
        total = torch.zeros((), device=self.config.device)
        count = 0

        for batch_x, batch_y in tqdm.tqdm(
            loader, desc="Training", leave=False, disable=not self.verbose
        ):
            x = batch_x.to(self.config.device)
            y = batch_y.to(self.config.device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
            schedule.before_step()
            optimizer.step()

            # Weighted by batch size, so a short last batch counts for what it holds
            total += loss.detach() * len(x)
            count += len(x)

        return total.item() / count

    def _validate(self, model, criterion, loader) -> float:
        return self._run_eval(model, criterion, loader, desc="Validation", show_progress=False)

    def _run_eval(self, model, criterion, loader, desc: str, show_progress: bool = True) -> float:
        model.eval()
        total = torch.zeros((), device=self.config.device)
        count = 0

        iterator = loader
        if show_progress:
            iterator = tqdm.tqdm(loader, desc=desc, leave=False, disable=not self.verbose)

        with torch.no_grad():
            for batch_x, batch_y in iterator:
                x = batch_x.to(self.config.device)
                y = batch_y.to(self.config.device)
                total += criterion(model(x), y) * len(x)
                count += len(x)

        return total.item() / count

    def _report(self, epoch: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(self.stage_name, epoch, self.config.epochs, message)
