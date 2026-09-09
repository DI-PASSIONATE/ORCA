import os

import numpy as np
import pandas as pd
import skrf as rf
import torch

from orca.logger import logger
from orca.training.codecs import OutputCodec
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.feature_transform import FeatureTransformPipeline
from orca.training.normalize import Normalizer
from orca.training.spec import FrequencyMode


class GeoToSParamDataset(BaseDataset):
    """
    The ORCA Dataset loads all .snp files from a specified directory and generates
    one training sample per geometry, covering the whole frequency band. Each sample
    consists of input parameters and the corresponding S-parameter curve.
    """

    frequency_mode = FrequencyMode.BAND

    def __init__(
        self,
        codec: OutputCodec,
        features: FeatureTransformPipeline | None = None,
        input_normalizer: Normalizer | None = None,
        output_normalizer: Normalizer | None = None,
    ):
        super(GeoToSParamDataset, self).__init__(
            codec, features, input_normalizer, output_normalizer
        )

    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        self.input_param_names = list(data_df.columns)
        self.input_param_names.remove("name")  # Remove 'name' column

        for idx, row in data_df.iterrows():
            snp_path = os.path.join(directory, row["name"])

            if not os.path.exists(snp_path):
                logger.error(f"S-parameter file not found: {snp_path}")
                continue

            geometry_params = np.array(row.drop("name"), dtype=np.float32)
            sample = self.load_single_sample(snp_path, geometry_params)
            self.samples.append(sample)

    def load_single_sample(
        self, sparam_path: str, geometry_params: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load S-parameter data from a Touchstone file and return a single sample with all frequencies."""
        net = rf.Network(sparam_path)

        if self.frequency_grid is None:
            self.frequency_grid = net.f

        # (n_freq, output_dim) -> (output_dim, n_freq), one row per output value
        y = self.codec.encode(net).T.astype(np.float32)

        # Input is only geometry parameters (no frequency)
        x = geometry_params.astype(np.float32)

        return (
            torch.tensor(x, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device),
        )
