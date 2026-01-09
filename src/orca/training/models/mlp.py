import torch
from typing_extensions import Self
import torch.nn as nn

class OrcaMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, input_mins, input_maxs):
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

        self.register_buffer("input_mins", torch.tensor(input_mins, dtype=torch.float32))
        self.register_buffer("input_maxs", torch.tensor(input_maxs, dtype=torch.float32))

        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.SiLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_dim))
        self.model = nn.Sequential(*layers)

    def normalize_minmax(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mins) / (self.input_maxs - self.input_mins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training data was normalized, so inference data must be normalized too
        x = self.normalize_minmax(x)

        # Training output was normalized too, but normalization is handled in the dataset class
        # and de-normalization is handled in the ONNX wrapper class.
        # This is so that training and loss function are always done on normalized data,
        # while for inference we can de-normalize after getting the output from the model 
        # and also export this denormalization to ONNX.
        return self.model(x)