import pandas as pd
import skrf as rf
import os
import numpy as np

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
    def __init__(self, data_dir: str, split: str = "all", n_ports=6, features=None, input_normalizer=None, output_normalizer=None):
        super(GeoToSParamDataset, self).__init__(data_dir, split, features, input_normalizer, output_normalizer)

        # Load parameters from CSV
        self.params_df = pd.read_csv(os.path.join(data_dir, "params.csv"))

        self.input_param_names = list(self.params_df.columns)
        self.input_param_names.remove('name')  # Remove 'name' column

        # Initialize output parameter names
        self.output_param_names = [
            f"S{i+1}{j+1}_{part}"
            for i in range(n_ports)
            for j in range(n_ports)
            for part in ("real", "imag")
        ]
    
        for idx, row in self.params_df.iterrows():
            geometry_name = row['name'] # = name.s4p
            s4p_path = f"{data_dir}/{geometry_name}_dc_deembedded.s{n_ports}p"

            if not os.path.exists(s4p_path):
                logger.error(f"S-parameter file not found: {s4p_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            sample = self.load_samples(s4p_path, geometry_params)

            rand_num = self.random.rand()
            if \
            (self.split == "all") or \
            (self.split == "train" and rand_num < 0.7) or \
            (self.split == "val" and 0.7 <= rand_num < 0.85) or \
            (self.split == "test" and rand_num >= 0.85):
                self.samples.append(sample)

        if self.output_normalizer is not None:
            self.output_normalizer.set_samples([y for _, y in self.samples])

    def load_samples(self, sparam_path: str, geometry_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        return (x, y)