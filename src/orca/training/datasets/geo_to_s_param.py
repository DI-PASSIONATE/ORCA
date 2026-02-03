import pandas as pd
import skrf as rf
import os
import numpy as np
import torch

from orca.training.datasets.base_dataset import BaseDataset
from orca.training.normalize import StandardNormalizer
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger

class GeoToSParamDataset(BaseDataset):
    """
    The ORCA Dataset loads all .snp files from a specified directory and generates
    a training sample for each frequency point in the S-parameter data. Each sample consists
    of input parameters and corresponding S-parameter values.
    """
    def __init__(self, n_ports=6, features=None, input_normalizer=None, output_normalizer=None):
        super(GeoToSParamDataset, self).__init__(features, input_normalizer, output_normalizer)

        self.n_ports = n_ports

        # Initialize output parameter names
        self.output_param_names = [
            f"S{i+1}{j+1}_{part}"
            for i in range(n_ports)
            for j in range(n_ports)
            for part in ("real", "imag")
        ]

    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        for idx, row in data_df.iterrows():
            geometry_name = row['name'] # = name.s4p
            snp_path = f"{directory}/{geometry_name}_dc_deembedded.s{self.n_ports}p"

            if not os.path.exists(snp_path):
                logger.error(f"S-parameter file not found: {snp_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            sample = self.load_single_sample(snp_path, geometry_params)
            self.samples.append(sample)
    
    def load_single_sample(self, sparam_path: str, geometry_params: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Load S-parameter data from a Touchstone file and return a single sample with all frequencies."""
        
        net = rf.Network(sparam_path)
        freq = net.f
        s = net.s

        # Stack all S-parameters across all frequencies
        # Shape: (n_freq, 4, 4) -> (32, n_freq)
        all_sparam_data = []
        for i in range(len(freq)):
            sij = s[i]
            # Real and imaginary parts stacked: (4, 4, 2) -> (32,)
            y_freq = np.stack((sij.real, sij.imag), axis=-1).reshape(-1)
            all_sparam_data.append(y_freq)
        
        # Stack across frequencies to get shape (32, n_freq)
        y = np.column_stack(all_sparam_data).astype(np.float32)

        # Input is only geometry parameters (no frequency)
        x = geometry_params.astype(np.float32)

        x, y = torch.tensor(x, dtype=torch.float32, device=self.device), torch.tensor(y, dtype=torch.float32, device=self.device)

        return (x, y)