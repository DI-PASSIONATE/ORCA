import json
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.training.basis_expansion import BasisExpansion, get_basis_class
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.models.base_model import OrcaModel, get_model_class
from orca.training.trainer import Trainer, TrainingConfig
from orca.training.tuner import HyperparameterTuner

if TYPE_CHECKING:
    from orca.geometry.base_geometry import BaseGeometry
    from orca.pipeline.context import PipelineContext


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(
        self,
        model: str | type[OrcaModel] = "mlp",
        basis: str | type[BasisExpansion] | None = None,
        hyperparameters: dict[str, Any] | str | None = None,
        test_frac: float = 0.15,
        val_frac: float = 0.15,
        n_train_samples: int | None = None,
        n_fold_cv: int = 5,
        n_trials: int = 200,
        tuning_timeout: float | None = None,
        max_epochs: int = 100,
        tuning_max_epochs: int = 30,
        batch_sizes: list[int] | None = None,
        regularization: bool = False,
        seed: int = 11,
    ):
        """
        Initializes the ModelTrainer stage with the architecture to train and optional
        hyperparameters. By default, optuna tuning is used to search for optimal
        hyperparameters, but specific hyperparameters can be provided to override the search space.

        The model is chosen here rather than on the geometry, so swapping architectures
        never requires touching a geometry class.

        Args:
            model: An OrcaModel subclass or a registered model name (default: "mlp").
            basis: Optional basis expansion of the model inputs, as a
                BasisExpansion subclass or a registered name (e.g. "chebyshev").
            hyperparameters: Hyperparameters to train with, as a dict or as the path of a
                JSON file holding one, e.g. the <name>_hyperparameters.json an earlier run
                saved in its models folder. If None, they are tuned with optuna. Every run
                saves the hyperparameters it trained with to that file.
            test_frac: Fraction of the geometries held out for testing (default: 0.15).
            val_frac: Fraction of the remaining geometries used for validation (early
                stopping and checkpoint selection) in the final training (default: 0.15).
            n_train_samples: Optional limit on the number of geometries used for training
                and tuning.
            n_fold_cv: Number of folds for cross-validation during hyperparameter tuning (default: 5).
            n_trials: Number of optuna trials during hyperparameter tuning (default: 200).
            tuning_timeout: Stop starting new tuning trials after this many seconds.
                None runs all n_trials.
            max_epochs: Epoch limit of the final training; early stopping usually ends
                it sooner. An "epochs" entry in hyperparameters takes precedence.
            tuning_max_epochs: Epoch limit of each cross-validation fold during tuning.
            batch_sizes: Batch sizes the tuning searches, e.g. [1024, 2048, 4096]. None
                searches 32 to 512. Ignored when hyperparameters are given.
            regularization: Also tune the weight decay and the model's regularization
                (dropout for the MLP). Without it, AdamW's default weight decay and no
                dropout are used, unless hyperparameters set them.
            seed: Seed for the data splits, the tuner and the weight initialisation.
        """
        super().__init__(name="Model Trainer", index=4)
        self.model_cls = get_model_class(model)
        self.basis_cls = get_basis_class(basis) if basis is not None else None
        self.hyperparameters = hyperparameters
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.n_samples = n_train_samples
        self.n_fold_cv = n_fold_cv
        self.n_trials = n_trials
        self.tuning_timeout = tuning_timeout
        self.max_epochs = max_epochs
        self.tuning_max_epochs = tuning_max_epochs
        self.batch_sizes = batch_sizes
        self.regularization = regularization
        self.seed = seed

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        geometry: BaseGeometry = context.geometry
        result_dir = context.result_dir
        result_csv = context.result_csv

        if not os.path.exists(result_csv):
            logger.error(f"No result CSV file found for training at {result_csv}.")
            return context

        # Read before anything else, so a wrong path fails before data is loaded
        given_hyperparameters = self._given_hyperparameters()

        result_df = order_parameter_columns(pd.read_csv(result_csv), geometry)
        torch.manual_seed(self.seed)

        # Split into train_val and test by geometry (CSV row) - train_val is used for
        # n-fold-cross-validation during hyperparameter tuning
        train_val_df, test_df = train_test_split(
            result_df, test_size=self.test_frac, random_state=self.seed
        )

        if self.n_samples is not None and self.n_samples < len(train_val_df):
            train_val_df = train_val_df.head(self.n_samples)
            logger.info(f"Using only the first {self.n_samples} samples for training as specified in the ModelTrainer initialization.")

        # Split train_val_df into train and val for actual model training
        train_df, val_df = train_test_split(
            train_val_df, test_size=self.val_frac, random_state=self.seed
        )
        self._save_split(context, train_df, val_df, test_df)

        # Perform hyperparameter tuning if needed. The result stays local rather than
        # being stored on the stage, so re-running the same ModelTrainer tunes again
        # instead of silently reusing the previous run's best parameters.
        if given_hyperparameters is not None:
            hyperparameters = given_hyperparameters
            logger.info(f"Using provided hyperparameters for training: {hyperparameters}")
        else:
            hyperparameters = self._tune(geometry, result_dir, train_val_df)

        # The training split owns the normalization statistics; validation reuses them,
        # so no validation data leaks into the normalization of the training data.
        train_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_df, fit_normalizers=True
        )
        val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)
        geometry.dataset.clear_cache()
        self._check_compatibility(train_dataset)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        trainer = Trainer(
            config=TrainingConfig.from_hyperparameters({"epochs": self.max_epochs, **hyperparameters}),
            progress_callback=progress_callback,
            stage_name=self.name,
        )
        spec = train_dataset.io_spec
        basis = self.basis_cls.from_spec(spec, hyperparameters) if self.basis_cls else None

        # Reseeded so the final weights do not depend on how much randomness tuning used
        torch.manual_seed(self.seed)

        result = trainer.fit(
            model=self.model_cls.from_spec(spec, hyperparameters, basis),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        if not math.isfinite(result.best_loss):
            raise RuntimeError(
                "Training diverged: the validation loss was never finite. Lower the learning "
                f"rate or check the data for NaNs. Hyperparameters: {hyperparameters}"
            )

        with open(context.hyperparameters_json_path, "w") as f:
            json.dump(hyperparameters, f, indent=2)
        logger.info(f"Hyperparameters saved to {context.hyperparameters_json_path}.")

        context.trained_model = result.model
        context.dataset = train_dataset
        context.hyperparameters = hyperparameters
        context.final_val_loss = result.best_loss
        context.training_history = result.history
        context.test_df = test_df
        return context

    def _given_hyperparameters(self) -> dict[str, Any] | None:
        """The hyperparameters passed to the stage, read from their JSON file if a path."""
        if self.hyperparameters is None or isinstance(self.hyperparameters, dict):
            return self.hyperparameters
        path = os.path.expanduser(self.hyperparameters)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ModelTrainer(hyperparameters={self.hyperparameters!r}) is neither a dict nor "
                "an existing JSON file."
            )
        with open(path) as f:
            hyperparameters = json.load(f)
        if not isinstance(hyperparameters, dict):
            raise TypeError(f"{path} holds a {type(hyperparameters).__name__}, not an object.")
        logger.info(f"Read the hyperparameters from {path}.")
        return hyperparameters

    def _tune(
        self, geometry: "BaseGeometry", result_dir: str, train_val_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Search the hyperparameters with cross-validation over the train/validation rows.

        A method of its own so the tuning data, which can fill most of a GPU, is
        released before the final training loads its splits.
        """
        logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
        train_val_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_val_df, fit_normalizers=True
        )
        self._check_compatibility(train_val_dataset)
        tuner = HyperparameterTuner(
            model_cls=self.model_cls,
            dataset=train_val_dataset,
            basis_cls=self.basis_cls,
            n_fold_cv=self.n_fold_cv,
            n_trials=self.n_trials,
            seed=self.seed,
            timeout=self.tuning_timeout,
            max_epochs=self.tuning_max_epochs,
            batch_sizes=self.batch_sizes,
            regularization=self.regularization,
        )
        hyperparameters = tuner.tune()
        logger.info(f"Hyperparameter tuning completed. Best hyperparameters: {hyperparameters}")
        return hyperparameters

    def _save_split(
        self,
        context: "PipelineContext",
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Record which result file went into which split, so a ModelTester run in a later
        pipeline evaluates exactly the held-out geometries.
        """
        os.makedirs(context.model_dir, exist_ok=True)
        split = pd.concat(
            [
                pd.DataFrame({"name": df["name"], "split": label})
                for label, df in (("train", train_df), ("val", val_df), ("test", test_df))
            ]
        )
        split.to_csv(context.split_csv_path, index=False)
        logger.info(
            f"Split {len(train_df)} train / {len(val_df)} validation / {len(test_df)} test "
            f"geometries; recorded in {context.split_csv_path}."
        )

    def _check_compatibility(self, dataset: BaseDataset) -> None:
        """
        Verifies that the model and the dataset agree on how frequency is laid out,
        so a mismatch fails here rather than as a shape error mid-training.
        """
        dataset_mode = type(dataset).frequency_mode
        if not self.model_cls.accepts_frequency_mode(dataset_mode):
            raise ValueError(
                f"{self.model_cls.__name__} consumes {self.model_cls.frequency_mode.name} samples, "
                f"but {type(dataset).__name__} produces {dataset_mode.name}. "
                "Pick a dataset and a model that agree on the frequency layout."
            )


def order_parameter_columns(data_df: pd.DataFrame, geometry: "BaseGeometry") -> pd.DataFrame:
    """
    Reorder a parameter table to ``name`` followed by the geometry's input parameters.

    Datasets take their inputs in column order, while the input normalizer and the
    exported model follow the geometry's parameter order. A hand-made or merged table
    with its columns in another order would otherwise be scaled column by column with
    the wrong ranges, without any error.

    Raises:
        ValueError: If the table lacks one of the geometry's parameters or has extra columns.
    """
    expected = ["name", *geometry.input_parameter_iterator.input_names]
    missing = [column for column in expected if column not in data_df.columns]
    extra = [column for column in data_df.columns if column not in expected]
    if missing or extra:
        raise ValueError(
            f"The parameter table does not match {geometry.name}'s input parameters: "
            f"missing {missing}, unexpected {extra}. Expected the columns {expected}."
        )
    return data_df[expected]
