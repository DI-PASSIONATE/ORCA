import torch
import torch.nn as nn
import numpy as np


class OrcaMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        output_shape: tuple[int, ...] | None = None,
    ):
        """
        Simple Multi-Layer Perceptron (MLP) model for regression tasks.

        Args:
            input_dim (int): Number of input parameters.
            hidden_sizes (list[int]): Sizes of hidden layers.
            output_dim (int): Total number of output parameters (flattened).
            output_shape (tuple[int, ...] | None):
                If provided, reshape output to (B, *output_shape).
                Otherwise, return flat output (B, output_dim).
        """
        super().__init__()

        # Check if output_shape is even possible
        if output_shape is not None and np.prod(output_shape) != output_dim:
            raise ValueError(
                f"output_shape {output_shape} does not match output_dim={output_dim}"
            )

        self.output_shape = output_shape

        layers = []
        in_size = input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.GELU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        y = self.model(x)

        # Optional output reshaping
        if self.output_shape is not None:
            y = y.view(y.shape[0], *self.output_shape)

        return y
