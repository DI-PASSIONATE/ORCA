import os
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
import skrf as rf
import torch

from orca.training.codecs import OutputCodec
from orca.training.normalize import Normalizer
from orca.training.spec import FrequencyMode, IOSpec


class BaseDataset(ABC, torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    #: Frequency layout of the samples this dataset produces.
    frequency_mode: ClassVar[FrequencyMode]

    def __init__(
        self,
        codec: OutputCodec,
        input_normalizer: Normalizer | None = None,
        output_normalizer: Normalizer | None = None,
    ):
        """
        Defines the base dataset class for ORCA training datasets.
        This class should be extended by specific dataset implementations.
        The output normalization is handled here, while input normalization is handled in the model itself.

        Args:
            codec (OutputCodec): Output representation used to encode targets.
            input_normalizer (Normalizer|None): Normalizer for input parameters.
            output_normalizer (Normalizer|None): Normalizer for output parameters.
        """
        super().__init__()

        self.codec = codec
        # Filled by load_samples, one entry per sample: the samples themselves, and the
        # result file each one came from. The groups keep cross-validation from putting
        # frequency points of one geometry on both sides of a fold.
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.sample_groups: list[str] = []
        # The normalized samples, stacked once loading is done
        self.inputs = torch.empty(0)
        self.targets = torch.empty(0)
        # Parsed Touchstone files, shared by every split made with new_split()
        self._touchstone_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.input_param_names: list[str] = []
        self.frequency_grid: np.ndarray | None = None
        self.random = np.random.RandomState(
            seed=11
        )  # Ensure same behavior for all instances
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Samples are built on self.device, and the predictor and ONNX exporter
        # apply the normalizers to tensors on that device too. Normalizers that
        # fit their buffers from the samples land there anyway; ones built from
        # declared ranges (MinMaxNormalizer) start on the CPU and are moved here.
        for normalizer in (self.input_normalizer, self.output_normalizer):
            if normalizer is not None:
                normalizer.to(self.device)

    @property
    def n_ports(self) -> int:
        return self.codec.n_ports

    @property
    def output_param_names(self) -> list[str]:
        return self.codec.output_names

    def read_touchstone(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        The frequencies and codec-encoded targets of a Touchstone file.

        Parsing dominates loading, and a training run reads most files twice: once
        for tuning and once for the final train/validation split. The result is kept
        until :meth:`clear_cache`, keyed by path and modification time.

        Args:
            path (str): Touchstone file to read.

        Returns:
            tuple[np.ndarray, np.ndarray]: Frequencies in Hz, shape ``(n_freq,)``, and
            targets of shape ``(n_freq, output_dim)``.
        """
        key = (path, os.stat(path).st_mtime_ns)
        cached = self._touchstone_cache.get(key)
        if cached is None:
            ntwk = rf.Network(path)
            cached = (ntwk.f, self.codec.encode(ntwk))
            self._touchstone_cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        """Forget the parsed Touchstone files, for this dataset and every split of it."""
        self._touchstone_cache.clear()

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All normalized inputs and targets, stacked along the first dimension.
        """
        return self.inputs, self.targets

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
            frequency_grid=self.frequency_grid,
        )

    def _load_samples_and_normalize(
        self, directory: str, data_df: pd.DataFrame, fit_normalizers: bool = False
    ) -> None:
        """
        Load the samples of a split and apply normalization to them.

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

        if len(self.sample_groups) != len(self.samples):
            raise RuntimeError(
                f"{type(self).__name__}.load_samples recorded {len(self.sample_groups)} sample "
                f"groups for {len(self.samples)} samples. Append the result file name of every "
                "sample to self.sample_groups, so cross-validation can split by geometry."
            )
        self._check_input_names()

        inputs, outputs = zip(*self.samples, strict=True)

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

        # Normalize the stacked tensors if normalizers are available.
        self.inputs = torch.stack(inputs)
        self.targets = torch.stack(outputs)
        self.samples = []
        if self.input_normalizer is not None:
            self.inputs = self.input_normalizer.normalize(self.inputs)
        if self.output_normalizer is not None:
            self.targets = self.output_normalizer.normalize(self.targets)

    def _check_input_names(self) -> None:
        """
        Fail if the input normalizer expects its columns in another order than the
        parameter table supplies them; it would otherwise scale every column with the
        range of a different one.
        """
        expected = getattr(self.input_normalizer, "input_names", None)
        if expected is not None and list(expected) != self.input_param_names:
            raise ValueError(
                f"The input normalizer expects the inputs {list(expected)}, but the "
                f"parameter table gives {self.input_param_names}. Reorder the table's "
                "columns to the geometry's input parameters (frequency last for per-point "
                "datasets), or declare the frequency band on the input parameter iterator."
            )

    @abstractmethod
    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        """
        Load samples from the dataset.

        Implementations append ``(input, target)`` tensor pairs to ``self.samples`` and,
        for each of them, the ``name`` of the table row it came from to
        ``self.sample_groups``. They set ``self.input_param_names`` as well.
        """

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)

    def new_split(
        self, directory: str, data_df: pd.DataFrame, fit_normalizers: bool = False
    ) -> "BaseDataset":
        """
        Create a new dataset split (train/val/test) with the same codec and
        normalizers. The new split will load its own samples from the provided data_df.

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
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
        )
        new_dataset._touchstone_cache = self._touchstone_cache  # noqa: SLF001 - same class
        new_dataset._load_samples_and_normalize(directory, data_df, fit_normalizers)  # noqa: SLF001 - same class
        return new_dataset
