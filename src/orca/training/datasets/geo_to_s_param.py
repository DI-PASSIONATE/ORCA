from torch.utils.data import Dataset
import pandas as pd
import skrf as rf
import os
import numpy as np
import torch

from orca.logger import logger

class GeoToSParamDataset(Dataset):
    """
    The ORCA Dataset loads all .snp files from a specified directory and generates
    a training sample for each frequency point in the S-parameter data. Each sample consists
    of input parameters and corresponding S-parameter values.
    """


    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples: list[tuple] = []

        # Load parameters from CSV
        self.params_df = pd.read_csv(os.path.join(data_dir, "params.csv"))

        self.input_param_names = list(self.params_df.columns) + ['frequency [100GHz]']
        self.input_param_names.remove('name')  # Remove 'name' column

        self.output_param_names = [
            'S11_real', 'S11_imag',
            'S21_real', 'S21_imag',
            'S31_real', 'S31_imag',
            'S41_real', 'S41_imag',
        ]

        for idx, row in self.params_df.iterrows():
            geometry_name = row['name'] # = name.s4p
            s4p_path = f"{data_dir}/{geometry_name}.s4p"

            if not os.path.exists(s4p_path):
                logger.error(f"S-parameter file not found: {s4p_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            samples = self.load_samples(f"{geometry_name}.s4p", geometry_params)
            self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def load_samples(self, filename: str, geometry_params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load S-parameter data from a Touchstone file."""
        samples = []
        sparam_path = f"{self.data_dir}/{filename}"
        net = rf.Network(sparam_path)
        freq = net.f

        # Scale frequency to multiple of 100 GHz
        freq = freq / 1e11  # Now in units of 100 GHz

        s = net.s

        for i in range(len(freq)):
            f = freq[i]
            sij = s[i]

            # Predict S11, S21, S31, S41 (Re/Im)
            y = np.array([
                np.real(sij[0, 0]), np.imag(sij[0, 0]),
                np.real(sij[1, 0]), np.imag(sij[1, 0]),
                np.real(sij[2, 0]), np.imag(sij[2, 0]),
                np.real(sij[3, 0]), np.imag(sij[3, 0]),
            ], dtype=np.float32)

            # Input = geometry + frequency
            x = np.hstack([geometry_params, f])

            samples.append((x, y))

        return samples