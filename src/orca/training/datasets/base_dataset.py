import torch
import numpy as np

from orca.geometry.base_geometry import BaseGeometry
from orca.training.normalize import Normalizer


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,  data_dir: str, geometry: BaseGeometry, split: str = "all", frequency_as_input: bool = False):
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
        self.split = split
        self.frequency_as_input = frequency_as_input
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []  # List of (input_params, output_params) tuples
        self.output_normalizer: Normalizer | None = None
        self.random = np.random.RandomState(seed=11)  # Ensure same behavior for all instances
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_train_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the training split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            geometry=self.geometry,
            split="train",
            frequency_as_input=self.frequency_as_input
        )
    
    def get_val_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the validation split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            geometry=self.geometry,
            split="val",
            frequency_as_input=self.frequency_as_input
        )
    
    def get_test_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the test split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            geometry=self.geometry,
            split="test",
            frequency_as_input=self.frequency_as_input
        )

    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]

        # Convert to tensors
        x, y = torch.tensor(x, dtype=torch.float32, device=self.device), torch.tensor(y, dtype=torch.float32, device=self.device)

        # Normalize output (input is normalized in the model itself, to embed it into the exported ONNX model)
        # Output is denormalized in the ONNX wrapper class after inference
        if self.output_normalizer is not None:
            y = self.output_normalizer(y)

        return x, y

    def __len__(self):
        return len(self.samples)