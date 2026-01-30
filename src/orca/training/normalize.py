import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

from orca.geometry.input_parameters import InputParameterIterator
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

    @abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize output parameters back to original scale.

        Args:
            x (torch.Tensor): Normalized tensor of shape (batch_size, n_output_parameters).
        Returns:
            torch.Tensor: Denormalized tensor of shape (batch_size, n_output_parameters).
        """
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for the forward method to normalize input parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input_parameters).
        Returns:
            torch.Tensor: Normalized tensor of shape (batch_size, n_input_parameters).
        """
        return self.forward(x)
    
class InputNormalizer(Normalizer):
    """
    A normalizer that applies normalization to input parameters.
    """
    @abstractmethod
    def __init__(self, input_parameter_iterator: InputParameterIterator, features: FeatureTransformPipeline | None = None):
        pass

class OutputNormalizer(Normalizer):
    """
    A normalizer that applies normalization to output parameters.
    """
    def set_samples(self, samples: list):
        """
        Sets the samples used to compute normalization statistics.
        Must be called before using the normalizer.

        Args:
            samples (list): List of output parameter samples.
        """
        # Makes sure samples are only set once
        if not hasattr(self, 'samples'):
            self.samples = samples
            self.process_samples(samples)

    @abstractmethod
    def process_samples(self, samples: list):
        """
        Process the samples to compute normalization statistics.

        Args:
            samples (list): List of output parameter samples.
        """
        pass

class MinMaxNormalizer(InputNormalizer):
    """
    A normalizer that applies min-max normalization to input parameters.
    """
    def __init__(self, input_parameter_iterator: InputParameterIterator, features: FeatureTransformPipeline | None = None):
        """
        Applies component-wise min-max normalization to the input tensor.

        Args:
            input_mins (list of float): Minimum values for each input parameter.
            input_maxs (list of float): Maximum values for each input parameter.
        """
        super(InputNormalizer, self).__init__()

        input_mins, input_maxs = input_parameter_iterator.get_min_max_values()

        if features is not None:
            input_mins, input_maxs = features.transform_min_max(input_mins, input_maxs) 

        self.register_buffer("input_mins", torch.tensor(input_mins, dtype=torch.float32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        self.register_buffer("input_maxs", torch.tensor(input_maxs, dtype=torch.float32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mins) / (self.input_maxs - self.input_mins)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.input_maxs - self.input_mins) + self.input_mins

class OutputMinMaxNormalizer(OutputNormalizer):
    """
    A normalizer that applies min-max normalization to output parameters.
    """
    def process_samples(self, samples: list[tuple[np.ndarray, np.ndarray]]):
        output_mins, output_maxs = self.get_output_min_max(samples)
        self.register_buffer("output_mins", output_mins)
        self.register_buffer("output_maxs", output_maxs)
    
    def get_output_min_max(self, samples) -> tuple[list[float], list[float]]:
        """Calculate min and max of output parameters for normalization."""
        stacked_samples = torch.vstack(samples)
        mins = torch.min(stacked_samples, dim=0).values
        maxs = torch.max(stacked_samples, dim=0).values
        print("OutputMinMaxNormalizer mins:", mins)
        print("OutputMinMaxNormalizer maxs:", maxs)
        return mins, maxs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.output_mins) / (self.output_maxs - self.output_mins + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins

class StandardNormalizer(OutputNormalizer):
    """
    A normalizer that applies standard score normalization to input parameters.
    """
    def process_samples(self, samples: list[tuple[np.ndarray, np.ndarray]]):
        input_means, input_stds = self.get_output_means_stds(samples)
        self.register_buffer("input_means", input_means)
        self.register_buffer("input_stds", input_stds)
    
    def get_output_means_stds(self, samples) -> tuple[list[float], list[float]]:
        """Calculate means and standard deviations of output parameters for normalization."""
        stacked_samples = torch.vstack(samples)
        means = torch.mean(stacked_samples, dim=0)
        stds = torch.std(stacked_samples, dim=0)
        print("StandardNormalizer means:", means)
        print("StandardNormalizer stds:", stds)
        return means, stds
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_means) / (self.input_stds + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.input_stds + 1e-8) + self.input_means