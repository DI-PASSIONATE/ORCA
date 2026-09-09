from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from orca.training.codecs import OutputCodec
from orca.training.feature_transform import FeatureTransformPipeline
from orca.training.normalize import Normalizer
from orca.training.spec import FrequencyMode, IOSpec


class BaseDataset(ABC, torch.utils.data.Dataset):
    #: Frequency layout of the samples this dataset produces.
    frequency_mode: ClassVar[FrequencyMode]

    def __init__(
        self,
        codec: OutputCodec,
        features: FeatureTransformPipeline | None = None,
        input_normalizer: Normalizer | None = None,
        output_normalizer: Normalizer | None = None,
    ):
        """
        Defines the base dataset class for ORCA training datasets.
        This class should be extended by specific dataset implementations.
        The output normalization is handled here, while input normalization is handled in the model itself.

        Args:
            codec (OutputCodec): Output representation used to encode targets.
            features (FeatureTransformPipeline | None): Optional feature pipeline.
            input_normalizer (Normalizer|None): Normalizer for input parameters.
            output_normalizer (Normalizer|None): Normalizer for output parameters.
        """
        super(BaseDataset, self).__init__()

        self.codec = codec
        self.features = features
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.input_param_names: list[str] = []
        self.frequency_grid: np.ndarray | None = None
        self.random = np.random.RandomState(
            seed=11
        )  # Ensure same behavior for all instances
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def n_ports(self) -> int:
        return self.codec.n_ports

    @property
    def output_param_names(self) -> list[str]:
        return self.codec.output_names

    @property
    def io_spec(self) -> IOSpec:
        """
        The shape contract models are built against.

        Only available once samples have been loaded, since the input parameter
        names are read from the result dataframe.

        Returns:
            IOSpec: Input names, feature count, output codec and frequency layout.
        """
        if not self.input_param_names:
            raise RuntimeError(
                "IOSpec is only available after samples have been loaded. "
                "Call new_split() or _load_samples_and_normalize() first."
            )
        return IOSpec(
            input_names=tuple(self.input_param_names),
            codec=self.codec,
            frequency_mode=type(self).frequency_mode,
            n_extra_features=len(self.features) if self.features is not None else 0,
            frequency_grid=self.frequency_grid,
        )

    def _load_samples_and_normalize(
        self, directory: str, data_df: pd.DataFrame, fit_normalizers: bool = False
    ) -> None:
        """
        Load features and apply normalization to the dataset samples.
        This method should be called after loading samples.

        Args:
            directory (str): Directory containing the Touchstone files.
            data_df (pd.DataFrame): Parameter table describing them.
            fit_normalizers (bool): Whether this split owns the normalization
                statistics. Exactly one split per training run should set this;
                the others reuse what it fitted, so no validation or test data
                leaks into the statistics.
        """
        self.load_samples(directory, data_df)

        if not self.samples:
            raise ValueError(
                f"No samples could be loaded from {directory}: none of the {len(data_df)} "
                "files listed in the parameter table exist. Check that the result "
                "directory and the CSV describe the same set of simulations."
            )

        if self.features is not None:
            self.samples = list(
                map(lambda s: (self.features(s[0]), s[1]), self.samples)
            )

        inputs, outputs = zip(*self.samples)

        # The normalizers are shared across the splits of a run, so only the split
        # that owns them fits; the rest reuse those statistics unchanged.
        for normalizer, split_samples in (
            (self.input_normalizer, inputs),
            (self.output_normalizer, outputs),
        ):
            if normalizer is None:
                continue
            if fit_normalizers:
                normalizer.fit(list(split_samples))
            elif not normalizer.is_fitted:
                raise RuntimeError(
                    f"{type(normalizer).__name__} has not been fitted. Load the "
                    "training split first with new_split(..., fit_normalizers=True)."
                )

        self.samples = list(
            map(
                lambda s: (
                    self.input_normalizer.normalize(s[0])
                    if self.input_normalizer is not None
                    else s[0],
                    self.output_normalizer.normalize(s[1])
                    if self.output_normalizer is not None
                    else s[1],
                ),
                self.samples,
            )
        )

    @abstractmethod
    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        """
        Load samples from the dataset.
        This method should be implemented by subclasses to load data specific from its self.data_dir.
        """
        pass

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def new_split(
        self, directory: str, data_df: pd.DataFrame, fit_normalizers: bool = False
    ) -> "BaseDataset":
        """
        Create a new dataset split (train/val/test) with the same codec, normalizers
        and feature pipeline. The new split will load its own samples from the provided data_df.

        Args:
            directory (str): Directory containing the dataset files for the new split.
            data_df (pd.DataFrame): DataFrame containing the data for the new split.
            fit_normalizers (bool): Whether this split should fit the shared
                normalizers. Pass True for the training split and False for the
                validation and test splits, which must reuse the training
                statistics.
        Returns:
            BaseDataset: New dataset split instance.
        """
        new_dataset = self.__class__(
            codec=self.codec,
            features=self.features,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
        )
        new_dataset._load_samples_and_normalize(directory, data_df, fit_normalizers)
        return new_dataset
