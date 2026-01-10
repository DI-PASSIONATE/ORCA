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
    def __init__(self, data_dir: str, geometry: BaseGeometry, split: str = "all", frequency_as_input: bool = True):
        super(GeoToSParamDataset, self).__init__(data_dir, geometry, split, frequency_as_input)

        # Load parameters from CSV
        self.params_df = pd.read_csv(os.path.join(data_dir, "params.csv"))

        self.input_param_names = list(self.params_df.columns) + ['frequency (GHz)']
        self.input_param_names.remove('name')  # Remove 'name' column

        self.output_param_names = [
            f"S{i+1}{j+1}_{part}"
            for i in range(4)
            for j in range(4)
            for part in ("real", "imag")
        ]
    
        for idx, row in self.params_df.iterrows():
            geometry_name = row['name'] # = name.s4p
            s4p_path = f"{data_dir}/{geometry_name}.s4p"

            if not os.path.exists(s4p_path):
                logger.error(f"S-parameter file not found: {s4p_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            samples = self.load_samples(f"{geometry_name}.s4p", geometry_params)

            rand_num = self.random.rand()
            if \
            (self.split == "all") or \
            (self.split == "train" and rand_num < 0.7) or \
            (self.split == "val" and 0.7 <= rand_num < 0.85) or \
            (self.split == "test" and rand_num >= 0.85):
                self.samples.extend(samples)

        self.output_normalizer = StandardNormalizer(self.samples)

    def load_samples(self, filename: str, geometry_params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Load S-parameter data from a Touchstone file."""
        samples = []
        sparam_path = f"{self.data_dir}/{filename}"
        
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

            samples.append((x, y))

        return samples