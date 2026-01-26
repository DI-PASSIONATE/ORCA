import torch
import numpy as np

from orca.training.normalize import Normalizer
from orca.training.feature_transform import FeatureTransformPipeline


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,  data_dir: str, split: str = "all", features: FeatureTransformPipeline | None = None, input_normalizer: Normalizer|None = None, output_normalizer: Normalizer|None = None):
        """
        Defines the base dataset class for ORCA training datasets.
        This class should be extended by specific dataset implementations.
        The output normalization is handled here, while input normalization is handled in the model itself.

        Args:
            data_dir (str): Directory containing the dataset files.
            split (str): Dataset split to use ('all', 'train', 'val', 'test').
            features (FeatureTransformPipeline | None): Optional feature pipeline.
            input_normalizer (Normalizer|None): Normalizer for input parameters.
            output_normalizer (Normalizer|None): Normalizer for output parameters.
        """
        super(BaseDataset, self).__init__()

        self.data_dir = data_dir
        #self.geometry = geometry
        self.split = split
        self.features = features
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []  # List of (input_params, output_params) tuples
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.random = np.random.RandomState(seed=11)  # Ensure same behavior for all instances
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def get_train_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the training split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            split="train",
            features=self.features,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
        )
    
    def get_val_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the validation split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            split="val",
            features=self.features,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
        )
    
    def get_test_split(self) -> 'BaseDataset':
        """
        Returns a dataset instance for the test split.
        """
        return self.__class__(
            data_dir=self.data_dir,
            split="test",
            features=self.features,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
        )
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]

        # Convert to tensors
        x, y = torch.tensor(x, dtype=torch.float32, device=self.device), torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if self.features is not None:
            x = self.features(x)

        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Normalize output
        # Output is denormalized in the ONNX wrapper class after inference
        if self.output_normalizer is not None:
            y = self.output_normalizer(y)

        return x, y

    def __len__(self):
        return len(self.samples)