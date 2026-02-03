import pandas as pd
import skrf as rf
import os
import numpy as np
import torch
import tqdm

from orca.training.datasets.base_dataset import BaseDataset
from orca.training.normalize import StandardNormalizer, MinMaxNormalizer, Normalizer
from orca.geometry.base_geometry import BaseGeometry
from orca.training.feature_transform import FeatureTransformPipeline
from orca.logger import logger

class GeoToSParamDatasetSingleFrequency(BaseDataset):
    """
    The ORCA Dataset loads all .snp files from a specified directory and generates
    a training sample for each frequency point in the S-parameter data. Each sample consists
    of input parameters and corresponding S-parameter values.
    """
    def __init__(self, features: FeatureTransformPipeline | None = None, n_ports: int = 6, input_normalizer: Normalizer|None = None, output_normalizer: Normalizer|None = None):
        super(GeoToSParamDatasetSingleFrequency, self).__init__(features, input_normalizer, output_normalizer)

        self.n_ports = n_ports

        self.output_param_names = [
            f"S{i+1}{j+1}_{part}"
            for i in range(self.n_ports)
            for j in range(self.n_ports)
            for part in ("real", "imag")
        ]

    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        self.input_param_names = list(data_df.columns) + ['frequency (GHz)']
        self.input_param_names.remove('name')  # Remove 'name' column
    
        for idx, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df), desc="Loading samples"):
            geometry_name = row['name'].replace(".gds", f"_dc_deembedded.s{self.n_ports}p")
            snp_path = os.path.join(directory, geometry_name)

            if not os.path.exists(snp_path):
                logger.debug(f"S-parameter file not found: {snp_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            samples = self.load_single_sample(snp_path, geometry_params)

            self.samples.extend(samples)

    def load_single_sample(self, sparam_path: str, geometry_params: np.ndarray) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Load S-parameter data from a Touchstone file."""
        samples = []
        
        net = rf.Network(sparam_path)
        freq = net.f
        s = net.s

        for i in range(len(freq)):
            f = freq[i]
            sij = s[i]

            # Predict ALL S-parameters for 4-port network
            y = np.stack((sij.real, sij.imag), axis=-1).reshape(-1).astype(np.float32)

            # Input = geometry + frequency
            x = np.hstack((geometry_params, f)).astype(np.float32)

            # Convert to tensors
            x, y = torch.tensor(x, dtype=torch.float32, device=self.device), torch.tensor(y, dtype=torch.float32, device=self.device)

            samples.append((x, y))

        return samples