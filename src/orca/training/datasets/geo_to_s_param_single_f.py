import os

import numpy as np
import pandas as pd
import skrf as rf
import torch
import tqdm

from orca.logger import logger
from orca.training.codecs import OutputCodec
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.feature_transform import FeatureTransformPipeline
from orca.training.normalize import Normalizer
from orca.training.spec import FrequencyMode


class GeoToSParamDatasetSingleFrequency(BaseDataset):
    """
    The ORCA Dataset loads all .snp files from a specified directory and generates
    a training sample for each frequency point in the S-parameter data. Each sample consists
    of input parameters and corresponding S-parameter values.
    """

    frequency_mode = FrequencyMode.PER_POINT

    def __init__(
        self,
        codec: OutputCodec,
        features: FeatureTransformPipeline | None = None,
        input_normalizer: Normalizer | None = None,
        output_normalizer: Normalizer | None = None,
    ):
        super(GeoToSParamDatasetSingleFrequency, self).__init__(
            codec, features, input_normalizer, output_normalizer
        )

    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        self.input_param_names = list(data_df.columns) + ["frequency"]
        self.input_param_names.remove("name")  # Remove 'name' column

        for idx, row in tqdm.tqdm(
            data_df.iterrows(), total=len(data_df), desc="Loading samples"
        ):
            snp_path = os.path.join(directory, row["name"])

            if not os.path.exists(snp_path):
                logger.warning(f"S-parameter file not found, skipping: {snp_path}")
                continue

            geometry_params = np.array(row.drop("name"), dtype=np.float32)
            samples = self.load_single_sample(snp_path, geometry_params)

            self.samples.extend(samples)

    def load_single_sample(
        self, sparam_path: str, geometry_params: np.ndarray
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Load S-parameter data from a Touchstone file, one sample per frequency point."""
        net = rf.Network(sparam_path)
        freq = net.f
        targets = self.codec.encode(net)  # (n_freq, output_dim)

        samples = []
        for i in range(len(freq)):
            # Input = geometry + frequency
            x = np.hstack((geometry_params, freq[i])).astype(np.float32)

            x, y = (
                torch.tensor(x, dtype=torch.float32, device=self.device),
                torch.tensor(targets[i], dtype=torch.float32, device=self.device),
            )

            samples.append((x, y))

        return samples
