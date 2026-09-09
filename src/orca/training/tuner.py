"""Hyperparameter search for ORCA surrogate models.

:class:`HyperparameterTuner` runs an optuna study over the union of the trainer's
own search space and the architecture search space the model class declares,
scoring each trial with k-fold cross-validation.
"""

from __future__ import annotations

import traceback
from typing import Any, Optional

import optuna
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset

from orca.logger import logger
from orca.training.models.base_model import OrcaModel
from orca.training.trainer import Trainer, TrainingConfig


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


class HyperparameterTuner:
    """Searches for the best hyperparameters of a model on a given dataset.

    Args:
        model_cls (type[OrcaModel]): Architecture to tune.
        dataset (Dataset): Train/validation data, split into folds internally.
            Must expose an ``io_spec`` (any :class:`~orca.training.datasets.base_dataset.BaseDataset`).
        n_fold_cv (int): Number of cross-validation folds per trial.
        n_trials (int): Number of optuna trials to run.
        seed (int): Seed for the fold split.
        sampler (optuna.samplers.BaseSampler | None): Defaults to TPE.
        pruner (optuna.pruners.BasePruner | None): Defaults to the median pruner.
    """

    def __init__(
        self,
        model_cls: type[OrcaModel],
        dataset: Dataset,
        n_fold_cv: int = 5,
        n_trials: int = 200,
        seed: int = 42,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        self.model_cls = model_cls
        self.dataset = dataset
        self.spec = dataset.io_spec
        self.n_fold_cv = n_fold_cv
        self.n_trials = n_trials
        self.seed = seed
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.pruner = pruner or optuna.pruners.MedianPruner()
        self.study: optuna.Study | None = None

    @property
    def search_space(self) -> dict[str, Any]:
        """Trainer hyperparameters merged with the model's architecture hyperparameters."""
        return {**TrainingConfig.search_space(), **self.model_cls.hyperparameter_search_space()}

    def tune(self) -> dict[str, Any]:
        """Run the study.

        Returns:
            dict[str, Any]: The best hyperparameters found.
        """
        self.study = optuna.create_study(
            direction="minimize", sampler=self.sampler, pruner=self.pruner
        )
        self.study.optimize(self._objective, n_trials=self.n_trials)
        return self.study.best_params

    def _objective(self, trial: optuna.Trial) -> float:
        hyperparameters = suggest_hyperparameters(trial, self.search_space)
        config = TrainingConfig.from_hyperparameters(hyperparameters)

        kfold = KFold(n_splits=self.n_fold_cv, shuffle=True, random_state=self.seed)
        fold_losses = []

        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(self.dataset)):
            fold_loss = self._run_fold(trial, config, hyperparameters, fold_idx, train_indices, val_indices)
            fold_losses.append(fold_loss)

        return sum(fold_losses) / len(fold_losses)

    def _run_fold(self, trial, config, hyperparameters, fold_idx, train_indices, val_indices) -> float:
        """Train one cross-validation fold, pruning the trial if anything fails."""
        fold_label = f"Fold {fold_idx + 1}/{self.n_fold_cv}"

        try:
            model = self.model_cls.from_spec(self.spec, hyperparameters)
            trainer = Trainer(
                config=config,
                stage_name=f"Tuning ({fold_label})",
                verbose=False,
            )
            result = trainer.fit(
                model=model,
                train_dataset=Subset(self.dataset, list(train_indices)),
                val_dataset=Subset(self.dataset, list(val_indices)),
            )
        except Exception:
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned()

        logger.info(f"{fold_label} | Val Loss: {result.best_loss:.4f}")

        # Report once per fold, using the fold index as the pruning step
        trial.report(result.best_loss, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return result.best_loss
