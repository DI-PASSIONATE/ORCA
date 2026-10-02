"""Hyperparameter search for ORCA surrogate models.

:class:`HyperparameterTuner` runs an optuna study over the union of three search
spaces - the trainer's own, the architecture's, and the basis expansion's - scoring
each trial with k-fold cross-validation over geometries.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from orca.logger import logger
from orca.training.trainer import EpochResult, Trainer, TrainingConfig

if TYPE_CHECKING:
    from orca.training.basis_expansion import BasisExpansion
    from orca.training.datasets.base_dataset import BaseDataset
    from orca.training.models.base_model import OrcaModel


def suggest_hyperparameters(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Draw one value per entry of a search space from an optuna trial.

    Entries may be a plain list of choices or any optuna distribution.

    Args:
        trial (optuna.Trial): The trial to sample from.
        search_space (dict[str, Any]): Names mapped to choices or distributions.

    Returns:
        dict[str, Any]: One concrete value per search space entry.
    """
    values: dict[str, Any] = {}
    for key, distribution in search_space.items():
        if isinstance(distribution, list):
            values[key] = trial.suggest_categorical(key, distribution)
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            values[key] = trial.suggest_int(
                key,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            values[key] = trial.suggest_float(
                key,
                distribution.low,
                distribution.high,
                step=distribution.step,
                log=distribution.log,
            )
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            values[key] = trial.suggest_categorical(key, distribution.choices)
        else:
            raise TypeError(
                f"Search space entry '{key}' is a {type(distribution).__name__}, which is not "
                "a list or a supported optuna distribution."
            )
    return values


def group_kfold_indices(
    groups: list[str], n_splits: int, seed: int | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold split that keeps every group on one side of each fold.

    A per-point dataset holds hundreds of samples per geometry. Splitting those samples
    at random puts the frequency neighbours of nearly every validation sample into the
    training fold, so the score measures interpolation along frequency instead of
    generalisation to unseen geometries. Splitting the geometries avoids that.

    Args:
        groups (list[str]): Group label of each sample (its result file).
        n_splits (int): Number of folds.
        seed (int | None): Seed for shuffling the groups; None shuffles unseeded.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Train and validation sample indices per fold.
    """
    labels, sample_group = np.unique(np.asarray(groups), return_inverse=True)
    if len(labels) < n_splits:
        raise ValueError(
            f"Cross-validation with {n_splits} folds needs at least {n_splits} geometries, "
            f"but the tuning data holds {len(labels)}."
        )
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (
            np.flatnonzero(np.isin(sample_group, train_groups)),
            np.flatnonzero(np.isin(sample_group, val_groups)),
        )
        for train_groups, val_groups in kfold.split(labels)
    ]


class HyperparameterTuner:
    """Searches for the best hyperparameters of a model on a given dataset.

    Args:
        model_cls (type[OrcaModel]): Architecture to tune.
        dataset (BaseDataset): Train/validation data, split into folds internally;
            its ``io_spec`` sizes the models.
        basis_cls (type[BasisExpansion] | None): Basis expansion to build each model
            with. Its search space is tuned alongside the model's. ``None`` trains
            on the raw inputs.
        n_fold_cv (int): Number of cross-validation folds per trial.
        n_trials (int): Number of optuna trials to run.
        seed (int | None): Seed for the fold split and the default sampler; None leaves
            both unseeded.
        sampler (optuna.samplers.BaseSampler | None): Defaults to TPE, seeded with ``seed``.
        pruner (optuna.pruners.BasePruner | None): Defaults to a median pruner that
            compares trials after every epoch, at the same fold and epoch.
        timeout (float | None): End the study after this many seconds: no new trial
            starts, and a trial still running is pruned after its current epoch.
            ``None`` runs all ``n_trials``.
        max_epochs (int): Epoch limit per fold; early stopping usually ends a fold sooner.
        batch_sizes (list[int] | None): Batch sizes to search. ``None`` uses
            :data:`~orca.training.trainer.DEFAULT_BATCH_SIZES`.
        regularization (bool): Also search the weight decay and the architecture's
            regularization hyperparameters (dropout for the MLP).
        training_defaults (dict[str, Any] | None): Trainer settings every trial uses
            but that are not searched, such as ``lr_schedule`` or ``grad_clip_norm``
            (see :class:`~orca.training.trainer.TrainingConfig`).
    """

    def __init__(
        self,
        model_cls: type[OrcaModel],
        dataset: BaseDataset,
        basis_cls: type[BasisExpansion] | None = None,
        n_fold_cv: int = 5,
        n_trials: int = 200,
        seed: int | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        timeout: float | None = None,
        max_epochs: int = 30,
        batch_sizes: list[int] | None = None,
        regularization: bool = False,
        training_defaults: dict[str, Any] | None = None,
    ):
        self.model_cls = model_cls
        self.dataset = dataset
        self.basis_cls = basis_cls
        self.spec = dataset.io_spec
        self.n_fold_cv = n_fold_cv
        self.n_trials = n_trials
        self.seed = seed
        self.sampler = sampler or optuna.samplers.TPESampler(seed=seed)
        self.pruner = pruner or optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        self.timeout = timeout
        self.max_epochs = max_epochs
        self.batch_sizes = batch_sizes
        self.regularization = regularization
        self.training_defaults = training_defaults or {}
        self._deadline = math.inf
        self.study: optuna.Study | None = None
        # The folds depend only on the data, so every trial is scored on the same split
        self.folds = group_kfold_indices(dataset.sample_groups, n_fold_cv, seed)

    @property
    def search_space(self) -> dict[str, Any]:
        """Trainer, architecture and basis-expansion hyperparameters, merged into one space."""
        basis_space = (
            self.basis_cls.hyperparameter_search_space() if self.basis_cls else {}
        )
        regularization_space = (
            self.model_cls.regularization_search_space() if self.regularization else {}
        )
        return {
            **TrainingConfig.search_space(self.batch_sizes, self.regularization),
            **self.model_cls.hyperparameter_search_space(),
            **regularization_space,
            **basis_space,
        }

    def tune(self) -> dict[str, Any]:
        """Run the study.

        Returns:
            dict[str, Any]: The best hyperparameters found.
        """
        self.study = optuna.create_study(
            direction="minimize", sampler=self.sampler, pruner=self.pruner
        )
        # optuna's own timeout only applies between trials, and one trial of a large
        # model can take many minutes; the deadline is also checked after every epoch.
        if self.timeout is not None:
            self._deadline = time.monotonic() + self.timeout
        self.study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout)
        if not any(t.state is optuna.trial.TrialState.COMPLETE for t in self.study.trials):
            raise RuntimeError(
                f"None of the {len(self.study.trials)} tuning trials completed; each one "
                "diverged, ran out of memory or was still running at the tuning timeout. "
                "Narrow the search space (lower learning rates, smaller models), allow "
                "more time, or pass hyperparameters directly."
            )
        return self.study.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        hyperparameters = suggest_hyperparameters(trial, self.search_space)
        config = TrainingConfig.from_hyperparameters(
            {**self.training_defaults, **hyperparameters}, epochs=self.max_epochs
        )
        fold_losses = [
            self._run_fold(trial, config, hyperparameters, fold_idx, train_indices, val_indices)
            for fold_idx, (train_indices, val_indices) in enumerate(self.folds)
        ]
        return sum(fold_losses) / len(fold_losses)

    def _run_fold(
        self, trial, config, hyperparameters, fold_idx, train_indices, val_indices
    ) -> float:
        """Train one cross-validation fold, letting the pruner stop it after any epoch.

        A trial whose training diverges or runs out of memory is pruned: those are
        properties of the hyperparameters. Any other exception is a bug and propagates.
        """
        fold_label = f"Fold {fold_idx + 1}/{self.n_fold_cv}"
        best_loss = math.inf

        def report(epoch: EpochResult) -> None:
            # The step counts epochs across folds, so the pruner compares a trial with
            # the others at the same fold and epoch; the best loss so far is reported,
            # since that is what the fold will score.
            nonlocal best_loss
            if time.monotonic() > self._deadline:
                logger.info(f"Tuning timeout reached during {fold_label}; pruning the trial.")
                raise optuna.exceptions.TrialPruned
            best_loss = min(best_loss, epoch.val_loss)
            if math.isfinite(best_loss):
                trial.report(best_loss, fold_idx * self.max_epochs + epoch.epoch - 1)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned

        basis = self.basis_cls.from_spec(self.spec, hyperparameters) if self.basis_cls else None
        model = self.model_cls.from_spec(self.spec, hyperparameters, basis)
        trainer = Trainer(config=config, stage_name=f"Tuning ({fold_label})", verbose=False)
        try:
            result = trainer.fit(
                model=model,
                train_dataset=Subset(self.dataset, train_indices.tolist()),
                val_dataset=Subset(self.dataset, val_indices.tolist()),
                epoch_callback=report,
            )
        except (torch.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"{fold_label} ran out of memory with {hyperparameters}; pruning.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned from e

        if not math.isfinite(result.best_loss):
            logger.warning(f"{fold_label} diverged with {hyperparameters}; pruning.")
            raise optuna.exceptions.TrialPruned

        logger.info(f"{fold_label} | Val Loss: {result.best_loss:.4f}")
        return result.best_loss
