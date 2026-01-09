import torch.nn as nn
import torch
import numpy as np

from orca.geometry.base_geometry import BaseGeometry
from orca.training.normalize import Normalizer, StandardNormalizer

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,  data_dir: str, geometry: BaseGeometry, frequency_as_input: bool = False):
        """
        Defines the base dataset class for ORCA training datasets.
        This class should be extended by specific dataset implementations.
        The output normalization is handled here, while input normalization is handled in the model itself.

        Args:
            data_dir (str): Directory containing the dataset files.
            geometry (BaseGeometry): Geometry object defining the input parameters.
            frequency_as_input (bool): Whether to include frequency as an input parameter.
        """
        super(BaseDataset, self).__init__()

        self.data_dir = data_dir
        self.geometry = geometry
        self.frequency_as_input = frequency_as_input
        self.samples: list[tuple] = [] # List of (input_params, output_params) tuples
        self.output_normalizer: Normalizer | None = None

    def get_output_means_stds(self) -> tuple[list[float], list[float]]:
        """Calculate means and standard deviations of output parameters for normalization."""
        all_outputs = np.array([y for _, y in self.samples], dtype=np.float32)
        means = np.mean(all_outputs, axis=0)
        stds = np.std(all_outputs, axis=0)
        return means, stds
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]

        # Convert to tensors
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        # Normalize output (input is normalized in the model itself, to embed it into the exported ONNX model)
        # Output is denormalized in the ONNX wrapper class after inference
        if self.output_normalizer is not None:
            y = self.output_normalizer(y)

        return x, y

    def __len__(self):
        return len(self.samples)