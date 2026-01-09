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
            input_mins.append(min_val)
            input_maxs.append(max_val)
        return input_mins, input_maxs