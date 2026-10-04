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
from orca.training.losses import SParameterLoss, srf_sample_weights
from orca.training.models.base_model import OrcaModel, get_model_class
from orca.training.spec import FrequencyMode
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
        lr_schedule: str = "cosine",
        warmup_epochs: float = 1.0,
        grad_clip_norm: float | None = 1.0,
        allow_tf32: bool = False,
        admittance_weight: float = 0.0,
        passivity_weight: float = 0.0,
        above_srf_weight: float = 1.0,
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
            n_fold_cv: Number of folds for cross-validation during hyperparameter tuning
                (default: 5). 1 tunes on the train/validation split of the final training
                instead, one training per trial; enough with a few thousand geometries.
            n_trials: Number of optuna trials during hyperparameter tuning (default: 200).
            tuning_timeout: Stop starting new tuning trials after this many seconds.
                None runs all n_trials.
            max_epochs: Epoch limit of the final training; early stopping usually ends
                it sooner. An "epochs" entry in hyperparameters takes precedence.
            tuning_max_epochs: Epoch limit of each cross-validation fold (or of the one
                hold-out training) during tuning.
            batch_sizes: Batch sizes the tuning searches, e.g. [1024, 2048, 4096]. None
                searches 32 to 512. Ignored when hyperparameters are given.
            regularization: Also tune the weight decay and the model's regularization
                (dropout for the MLP). Without it, AdamW's default weight decay and no
                dropout are used, unless hyperparameters set them.
            lr_schedule: "cosine" decays the learning rate smoothly to 1% of its initial
                value over the epoch limit; "plateau" halves it whenever the validation
                loss stalls. Used by the tuning trials and the final training.
            warmup_epochs: Epochs over which the learning rate rises linearly to its
                tuned value at the start of every training; 0 disables the warmup.
            grad_clip_norm: Largest gradient norm of an optimizer step; larger gradients
                are scaled down. None disables clipping.
            allow_tf32: Train with TF32 matrix multiplies on GPUs that support them
                (Ampere and newer, e.g. A100): faster, with 10 instead of 23 mantissa
                bits in the multiplies. Tuning and the final training use it; testing and
                the exported model stay in full FP32.
            admittance_weight: Weight of a loss term on the relative error of the
                predicted admittance matrix Y, of Y as a whole and of its real part, which
                track L and Q far more closely than S does (see
                orca.training.losses.admittance_error). 0 (default) leaves it out;
                0.1 to 1 puts it on the scale of the S-parameter loss.
            passivity_weight: Weight of a penalty on predicted S-matrices whose largest
                singular value exceeds 1 (or the simulated one, where that is larger).
                0 (default) leaves it out.
            above_srf_weight: Loss weight of the frequency points above each geometry's
                first self-resonance, relative to the points below it; e.g. 0.1 spends
                the model's capacity on the band an inductor is used in. 1 (default)
                weights all points alike. The resonance is found in the simulated
                S-parameters (orca.training.losses.first_self_resonance).

        The three loss options need a per-point dataset. They change what the
        validation loss measures, so losses are only comparable between runs with the
        same options.
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
        self.training_defaults = {
            "lr_schedule": lr_schedule,
            "warmup_epochs": warmup_epochs,
            "grad_clip_norm": grad_clip_norm,
            "allow_tf32": allow_tf32,
        }
        self.admittance_weight = admittance_weight
        self.passivity_weight = passivity_weight
        self.above_srf_weight = above_srf_weight
        # Checked now, so a typo fails before hours of tuning rather than after
        TrainingConfig.from_hyperparameters(self.training_defaults)
        if admittance_weight < 0 or passivity_weight < 0:
            raise ValueError("admittance_weight and passivity_weight must not be negative.")
        if above_srf_weight <= 0:
            raise ValueError(f"above_srf_weight must be positive, got {above_srf_weight}.")

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
        # The run seed (ORCA.run(seed=...)) fixes the splits, the tuner and the weights
        seed = context.seed
        if seed is not None:
            torch.manual_seed(seed)

        # Split into train_val and test by geometry (CSV row) - train_val is used for
        # n-fold-cross-validation during hyperparameter tuning
        train_val_df, test_df = train_test_split(
            result_df, test_size=self.test_frac, random_state=seed
        )

        if self.n_samples is not None and self.n_samples < len(train_val_df):
            train_val_df = train_val_df.head(self.n_samples)
            logger.info(f"Using only the first {self.n_samples} samples for training as specified in the ModelTrainer initialization.")

        # Split train_val_df into train and val for actual model training
        train_df, val_df = train_test_split(
            train_val_df, test_size=self.val_frac, random_state=seed
        )
        self._save_split(context, train_df, val_df, test_df)

        # Perform hyperparameter tuning if needed. The result stays local rather than
        # being stored on the stage, so re-running the same ModelTrainer tunes again
        # instead of silently reusing the previous run's best parameters.
        if given_hyperparameters is not None:
            hyperparameters = given_hyperparameters
            logger.info(f"Using provided hyperparameters for training: {hyperparameters}")
        else:
            hyperparameters = self._tune(geometry, result_dir, train_val_df, val_df, seed)

        # The training split owns the normalization statistics; validation reuses them,
        # so no validation data leaks into the normalization of the training data.
        train_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_df, fit_normalizers=True
        )
        val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)
        geometry.dataset.clear_cache()
        self._check_compatibility(train_dataset)
        criterion_factory = self._criterion_factory(train_dataset)
        self._weight_samples(train_dataset)
        self._weight_samples(val_dataset)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        spec = train_dataset.io_spec
        basis = self.basis_cls.from_spec(spec, hyperparameters) if self.basis_cls else None

        # Reseeded so the final weights do not depend on how much randomness tuning used
        if seed is not None:
            torch.manual_seed(seed)

        model = self.model_cls.from_spec(spec, hyperparameters, basis)
        trainer = Trainer(
            config=TrainingConfig.from_hyperparameters(
                {"epochs": self.max_epochs, **self.training_defaults, **hyperparameters}
            ),
            criterion=criterion_factory(model) if criterion_factory else None,
            progress_callback=progress_callback,
            stage_name=self.name,
        )
        result = trainer.fit(model=model, train_dataset=train_dataset, val_dataset=val_dataset)
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
        self,
        geometry: "BaseGeometry",
        result_dir: str,
        train_val_df: pd.DataFrame,
        val_df: pd.DataFrame,
        seed: int | None,
    ) -> dict[str, Any]:
        """
        Search the hyperparameters with cross-validation over the train/validation rows, or
        on the validation rows of the final training if `n_fold_cv` is 1.

        A method of its own so the tuning data, which can fill most of a GPU, is
        released before the final training loads its splits.
        """
        logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
        train_val_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_val_df, fit_normalizers=True
        )
        self._check_compatibility(train_val_dataset)
        self._weight_samples(train_val_dataset)
        tuner = HyperparameterTuner(
            model_cls=self.model_cls,
            dataset=train_val_dataset,
            basis_cls=self.basis_cls,
            n_fold_cv=self.n_fold_cv,
            n_trials=self.n_trials,
            seed=seed,
            timeout=self.tuning_timeout,
            max_epochs=self.tuning_max_epochs,
            batch_sizes=self.batch_sizes,
            regularization=self.regularization,
            training_defaults=self.training_defaults,
            # The dataset labels each sample with its result file, the name column of the table
            holdout_groups=set(val_df["name"]) if self.n_fold_cv == 1 else None,
            criterion_factory=self._criterion_factory(train_val_dataset),
        )
        hyperparameters = tuner.tune()
        logger.info(f"Hyperparameter tuning completed. Best hyperparameters: {hyperparameters}")
        return hyperparameters

    @property
    def _custom_loss(self) -> bool:
        """Whether any loss option asks for more than the model's default loss."""
        return self.admittance_weight > 0 or self.passivity_weight > 0 or self.above_srf_weight != 1

    def _criterion_factory(
        self, dataset: BaseDataset
    ) -> Callable[[OrcaModel], SParameterLoss] | None:
        """
        The loss of each model trained on ``dataset``'s outputs, or None for the model's
        default loss. A factory, since the data term is the default loss of the model.
        """
        if not self._custom_loss:
            return None
        if type(dataset).frequency_mode is not FrequencyMode.PER_POINT:
            raise ValueError(
                "ModelTrainer's admittance_weight, passivity_weight and above_srf_weight need "
                f"a per-point dataset, but {type(dataset).__name__} lays out frequency as "
                f"{type(dataset).frequency_mode.name}."
            )
        codec, output_normalizer = dataset.codec, dataset.output_normalizer

        def build(model: OrcaModel) -> SParameterLoss:
            return SParameterLoss(
                model.default_loss(),
                codec=codec,
                output_normalizer=output_normalizer,
                admittance_weight=self.admittance_weight,
                passivity_weight=self.passivity_weight,
            )

        return build

    def _weight_samples(self, dataset: BaseDataset) -> None:
        """Weight the frequency points above each geometry's self-resonance, if asked to."""
        if self.above_srf_weight != 1:
            dataset.set_sample_weights(srf_sample_weights(dataset, self.above_srf_weight))

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
