import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np


    
class Feature(nn.Module, ABC):
    """
    Abstract base class for feature transformation modules.
    This allows geometries to define custom features to be calculated from input parameters
    before being fed into the machine learning model.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute additional features from input parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input_parameters).
        Returns:
            torch.Tensor: Transformed features tensor of shape (batch_size, n_transformed_features).
        """

    @abstractmethod
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        """
        Calculate the minimum and maximum values of the transformed feature
        based on the provided input parameter min/max values.

        Args:
            input_mins (list of float): Minimum values for each input parameter.
            input_maxs (list of float): Maximum values for each input parameter.
        Returns:
            tuple: (min_value, max_value) of the transformed feature.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of new features added by this transform.
        """
        pass
    
class RatioFeature(Feature):
    """
    A feature transform that computes ratios of specified input parameters.
    """
    def __init__(self, i: int, j: int):
        super(RatioFeature, self).__init__()
        self.i = i
        self.j = j
        self.eps = 1e-8  # Small constant to avoid division by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add batch dimension if missing
        if len(x.shape) == 1:
            new_feature = x[self.i] / (x[self.j] + self.eps)
            return torch.cat([x, new_feature.unsqueeze(0)], dim=0)
        else:
            new_feature = x[:, self.i] / (x[:, self.j] + self.eps)
            return torch.cat([x, new_feature.unsqueeze(1)], dim=1)
    
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        min_i = input_mins[self.i]
        max_i = input_maxs[self.i]
        min_j = input_mins[self.j]
        max_j = input_maxs[self.j]

        possible_values = [
            min_i / (min_j + self.eps),
            min_i / (max_j + self.eps),
            max_i / (min_j + self.eps),
            max_i / (max_j + self.eps),
        ]
        return min(possible_values), max(possible_values)
    
    def __len__(self) -> int:
        return 1

    
class ChebyshevFeature(Feature):
    """
    A feature transform that computes Chebyshev polynomial features of specified degree
    for a given input parameter.
    """
    def __init__(self, i: int, degree: int, min_freq: float = 1e9, max_freq: float = 200e9):
        super(ChebyshevFeature, self).__init__()
        self.index = i
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is 1D or 2D (batched or single sample)
        output_shape = x.shape
        if len(output_shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        # Normalize the input parameter to the range [-1, 1]
        x_input = x[:, self.index] # One parameter over entire batch
        x_normalized = 2 * (x_input - self.min_freq) / (self.max_freq - self.min_freq) - 1
        T = [torch.ones_like(x_normalized)]
        if self.degree >= 1:
            T.append(x_normalized)
        for n in range(2, self.degree + 1):
            Tn = 2 * x_normalized * T[n - 1] - T[n - 2]
            T.append(Tn)
        # Convert to (batch_size, degree) shape
        new_features = torch.stack(T[1:], dim=1)
        # Return old + new features
        if len(new_features.shape) == 1:
            new_features = new_features.unsqueeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(1)
        res = torch.cat([x, new_features], dim=1)

        # If original input was 1D, remove batch dimension
        if len(output_shape) == 1:
            res = res.squeeze(0)
        return res
    
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        # Chebyshev polynomials oscillate between -1 and 1
        return -1.0, 1.0
    
    def __len__(self) -> int:
        return self.degree
    
class LogFeature(Feature):
    """
    A feature transform that computes the logarithm of a specified input parameter.
    """
    def __init__(self, i: int, scale: float = 1.0):
        super(LogFeature, self).__init__()
        self.index = i
        self.scale = scale
        self.eps = 1e-8  # Small constant to avoid log(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_feature = torch.log(x[:, self.index] / self.scale + self.eps)
        # Return old + new features
        res = torch.cat([x, log_feature.unsqueeze(1)], dim=1)
        return res
    
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        min_val = np.log(input_mins[self.index] / self.scale + self.eps)
        max_val = np.log(input_maxs[self.index] / self.scale + self.eps)
        return min_val, max_val
    
    def __len__(self) -> int:
        return 1
    
class SinFeature(Feature):
    """
    A feature transform that computes the sine of a specified input parameter.
    """
    def __init__(self, i: int):
        super(SinFeature, self).__init__()
        self.index = i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sin_feature = torch.sin(x[:, self.index])
        # Return old + new features
        res = torch.cat([x, sin_feature.unsqueeze(1)], dim=1)
        return res
    
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        # Sine function oscillates between -1 and 1
        return -1.0, 1.0
    
    def __len__(self) -> int:
        return 1
    
class CosFeature(Feature):
    """
    A feature transform that computes the cosine of a specified input parameter.
    """
    def __init__(self, i: int):
        super(CosFeature, self).__init__()
        self.index = i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos_feature = torch.cos(x[:, self.index])
        # Return old + new features
        res = torch.cat([x, cos_feature.unsqueeze(1)], dim=1)
        return res
    
    def calculate_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[float, float]:
        # Cosine function oscillates between -1 and 1
        return -1.0, 1.0
    
    def __len__(self) -> int:
        return 1

class FeatureTransformPipeline(nn.Module):
    """
    A pipeline to apply multiple feature transforms sequentially.
    """
    def __init__(self, *transforms: Feature):
        super(FeatureTransformPipeline, self).__init__()
        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)

        return x
    
    def transform_min_max(self, input_mins: list[float], input_maxs: list[float]) -> tuple[list[float], list[float]]:
        for transform in self.transforms:
            min_val, max_val = transform.calculate_min_max(input_mins, input_maxs)
            for _ in range(len(transform)):
                input_mins.append(min_val)
                input_maxs.append(max_val)
        return input_mins, input_maxs
    
    def __len__(self) -> int:
        # Return total number of new features added by all transforms
        total_new_features = 0
        for transform in self.transforms:
            total_new_features += len(transform)
        return total_new_features