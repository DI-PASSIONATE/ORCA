import pandas as pd
import skrf as rf
import os
import numpy as np
import torch
import tqdm

from orca.training.normalize import Normalizer
from orca.training.feature_transform import FeatureTransformPipeline
from orca.logger import logger


class GeoToNtwkDataset(torch.utils.data.Dataset):
    """
    This dataset class is designed to load a scikit-rf Network object from a .snp file and associate it with the corresponding geometry parameters. 
    This can be used in the testing stage to compare the predicted S-parameters with the actual S-parameters from the .snp file.
    """

    def __init__(
        self,
        directory: str,
        data_df: pd.DataFrame,
        n_ports: int = 6,
    ):
        self.n_ports = n_ports
        self.samples = []
        self.load_samples(directory, data_df)


    def load_samples(self, directory: str, data_df: pd.DataFrame) -> None:
        self.input_param_names = list(data_df.columns)
        self.input_param_names.remove("name")  # Remove 'name' column

        for idx, row in tqdm.tqdm(
            data_df.iterrows(), total=len(data_df), desc="Loading test network samples"
        ):
            geometry_name = row["name"].replace(
                ".gds", f"_dc_deembedded.s{self.n_ports}p"
            )
            snp_path = os.path.join(directory, geometry_name)

            if not os.path.exists(snp_path):
                logger.debug(f"S-parameter file not found: {snp_path}")
                continue

            geometry_params = np.array(row.drop("name"), dtype=np.float32)
            samples = self.load_single_sample(snp_path, geometry_params)

            self.samples.extend(samples)

    def load_single_sample(
        self, sparam_path: str, geometry_params: np.ndarray
    ) -> list[tuple[np.ndarray, rf.Network]]:
        """Load S-parameter data from a Touchstone file."""

        return [(geometry_params, rf.Network(sparam_path))]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]