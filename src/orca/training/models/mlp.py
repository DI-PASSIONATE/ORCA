import torch
import torch.nn as nn
from orca.training.normalize import Normalizer
from orca.training.feature_transform import FeatureTransformPipeline

class OrcaMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int, normalizer: Normalizer, features: FeatureTransformPipeline | None = None):
        """
        Simple Multi-Layer Perceptron (MLP) model for regression tasks.
        Args:
            input_dim (int): Amount of input parameters.
            hidden_sizes (list of int): List containing the sizes of hidden layers. The amount of hidden layers is determined by the length of this list.
            output_dim (int): Amount of output parameters.
            input_mins (list of float): Minimum values for each input parameter for normalization.
            input_maxs (list of float): Maximum values for each input parameter for normalization.
        """
        super(OrcaMLP, self).__init__()
        layers = []
        in_size = input_dim

        self.normalizer = normalizer
        self.features = features
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.Sigmoid())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate additional features if any
        if self.features is not None:
            x = self.features(x)

        # Training data was normalized, so inference data must be normalized too
        if self.normalizer is not None:
            x = self.normalizer(x)

        # Training output was normalized too, but normalization is handled in the dataset class
        # and de-normalization is handled in the ONNX wrapper class.
        # This is so that training and loss function are always done on normalized data,
        # while for inference we can de-normalize after getting the output from the model 
        # and also export this denormalization to ONNX.
        return self.model(x)