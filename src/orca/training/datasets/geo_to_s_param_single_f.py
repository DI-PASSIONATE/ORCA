import pandas as pd
import skrf as rf
import os
import numpy as np
import torch

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
    def __init__(self, data_dir: str, split: str = "all", features: FeatureTransformPipeline | None = None,n_ports: int = 6, input_normalizer: Normalizer|None = None, output_normalizer: Normalizer|None = None):
        super(GeoToSParamDatasetSingleFrequency, self).__init__(data_dir, split, features, input_normalizer, output_normalizer)

        # Load parameters from CSV
        self.params_df = pd.read_csv(os.path.join(data_dir, "params.csv"))

        self.input_param_names = list(self.params_df.columns) + ['frequency (GHz)']
        self.input_param_names.remove('name')  # Remove 'name' column

        self.output_param_names = [
            f"S{i+1}{j+1}_{part}"
            for i in range(n_ports)
            for j in range(n_ports)
            for part in ("real", "imag")
        ]
    
        for idx, row in self.params_df.iterrows():
            geometry_name = row['name'] # = name.s4p
            snp_path = f"{data_dir}/{geometry_name}_dc_deembedded.s6p"

            if not os.path.exists(snp_path):
                logger.warning(f"S-parameter file not found: {snp_path}")
                continue

            geometry_params = np.array(row.drop('name'), dtype=np.float32)
            samples = self.load_samples(f"{geometry_name}_dc_deembedded.s6p", geometry_params)

            rand_num = self.random.rand()
            if \
            (self.split == "all") or \
            (self.split == "train" and rand_num < 0.7) or \
            (self.split == "val" and 0.7 <= rand_num < 0.85) or \
            (self.split == "test" and rand_num >= 0.85):
                self.samples.extend(samples)

        print(f"Loaded {len(self.samples)} samples for split '{self.split}' from {data_dir}")

        if self.output_normalizer is not None:
            self.output_normalizer.set_samples([y for _, y in self.samples])

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