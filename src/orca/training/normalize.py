import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

from orca.training.feature_transform import FeatureTransformPipeline

class Normalizer(nn.Module, ABC):
    """
    Abstract base class for normalization modules.
    This allows geometries to define custom normalization strategies for input parameters
    before being fed into the machine learning model.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to normalize input parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input_parameters).
        Returns:
            torch.Tensor: Normalized tensor of shape (batch_size, n_input_parameters).
        """
        pass

    @abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize output parameters back to original scale.

        Args:
            x (torch.Tensor): Normalized tensor of shape (batch_size, n_output_parameters).
        Returns:
            torch.Tensor: Denormalized tensor of shape (batch_size, n_output_parameters).
        """
        pass

class MinMaxNormalizer(Normalizer):
    """
    A normalizer that applies min-max normalization to input parameters.
    """
    def __init__(self, input_mins: list[float], input_maxs: list[float], features: FeatureTransformPipeline | None = None):
        """
        Applies component-wise min-max normalization to the input tensor.

        Args:
            input_mins (list of float): Minimum values for each input parameter.
            input_maxs (list of float): Maximum values for each input parameter.
        """
        super(MinMaxNormalizer, self).__init__()

        if features is not None:
            print(f"Mins before feature transform: {input_mins}")
            print(f"Maxs before feature transform: {input_maxs}")
            input_mins, input_maxs = features.transform_min_max(input_mins, input_maxs) 
            print(f"Mins after feature transform: {input_mins}")
            print(f"Maxs after feature transform: {input_maxs}")

        self.register_buffer("input_mins", torch.tensor(input_mins, dtype=torch.float32))
        self.register_buffer("input_maxs", torch.tensor(input_maxs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mins) / (self.input_maxs - self.input_mins)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.input_maxs - self.input_mins) + self.input_mins
    
class StandardNormalizer(Normalizer):
    """
    A normalizer that applies standard score normalization to input parameters.
    """
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray]]):
        """
        Applies component-wise standard score normalization to the input tensor.
        Args:
            input_means (list[float]): Mean values for each input parameter.
            input_stds (list[float]): Standard deviation values for each input parameter.
        """
        super(StandardNormalizer, self).__init__()
        input_means, input_stds = self.get_output_means_stds(samples)
        self.register_buffer("input_means", torch.tensor(input_means, dtype=torch.float32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        self.register_buffer("input_stds", torch.tensor(input_stds, dtype=torch.float32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    def get_output_means_stds(self, samples) -> tuple[list[float], list[float]]:
        """Calculate means and standard deviations of output parameters for normalization."""
        all_outputs = np.array([y for _, y in samples], dtype=np.float32)
        means = np.mean(all_outputs, axis=0)
        stds = np.std(all_outputs, axis=0)
        return means, stds
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_means) / (self.input_stds + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.input_stds + 1e-8) + self.input_means