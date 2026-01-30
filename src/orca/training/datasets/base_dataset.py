import torch
import numpy as np
from abc import ABC, abstractmethod

from orca.training.normalize import Normalizer
from orca.training.feature_transform import FeatureTransformPipeline


class BaseDataset(ABC, torch.utils.data.Dataset):
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
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.random = np.random.RandomState(seed=11)  # Ensure same behavior for all instances
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_samples_and_normalize(self) -> None:
        """
        Load features and apply normalization to the dataset samples.
        This method should be called after loading samples.
        """
        self.load_samples()

        if self.features is not None:
            self.samples = list(map(lambda s: (self.features(s[0]), s[1]), self.samples))
        
        # Set the samples here. Since the normalizer is passed to all dataset splits,
        # and the mean/std should only be computed once, the normalizer checks if samples have already been set.
        # The first time set_samples is called (meaning for the 'train' split), it computes and stores the statistics.
        # This way, the 'val' and 'test' splits will use the same normalization parameters, but no data leakage occurs.
        inputs, outputs = zip(*self.samples)

        if self.input_normalizer is not None:
            self.input_normalizer.set_samples(list(inputs))
        if self.output_normalizer is not None:
            self.output_normalizer.set_samples(list(outputs))
            
        self.samples = list(map(lambda s: (
            self.input_normalizer.normalize(s[0]) if self.input_normalizer is not None else s[0],
            self.output_normalizer.normalize(s[1]) if self.output_normalizer is not None else s[1]
        ), self.samples))

    @abstractmethod
    def load_samples(self) -> None:
        """
        Load samples from the dataset.
        This method should be implemented by subclasses to load data specific from its self.data_dir.
        """
        pass

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
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)