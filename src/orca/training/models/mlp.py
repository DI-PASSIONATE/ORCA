from typing import Any

import optuna
import torch
import torch.nn as nn

from orca.training.basis_expansion import BasisExpansion
from orca.training.models.base_model import OrcaModel, register_model
from orca.training.spec import FrequencyMode, IOSpec


@register_model("mlp")
class OrcaMLP(OrcaModel):
    """Plain multi-layer perceptron over geometry parameters and frequency.

    One forward pass predicts the response at a single frequency point, so the
    frequency curve is assembled by batching the same geometry over the band.
    Nothing about the output is constrained: the network is free to predict a
    non-passive, non-reciprocal S-matrix.

    Args:
        spec (IOSpec): Dataset contract the model is sized against.
        hidden_sizes (list[int]): Width of each hidden layer.
        activation (type[nn.Module]): Activation module class used between layers.
        dropout (float): Dropout probability applied after each activation.
        basis (BasisExpansion | None): Optional basis expansion of the inputs,
            applied before the first layer. See :mod:`orca.training.basis_expansion`.
    """

    frequency_mode = FrequencyMode.PER_POINT

    def __init__(
        self,
        spec: IOSpec,
        hidden_sizes: list[int],
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        basis: BasisExpansion | None = None,
    ):
        super().__init__(spec, basis)

        layers: list[nn.Module] = []
        in_size = self.expanded_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, spec.output_dim))

        self.model = nn.Sequential(*layers)

    @classmethod
    def from_spec(
        cls,
        spec: IOSpec,
        hyperparameters: dict[str, Any],
        basis: BasisExpansion | None = None,
    ) -> "OrcaMLP":
        num_layers = hyperparameters.get("num_layers", 4)
        hidden_size = hyperparameters.get("hidden_size", 512)
        activation = getattr(nn, hyperparameters.get("activation_function", "GELU"))
        return cls(
            spec=spec,
            hidden_sizes=[hidden_size] * num_layers,
            activation=activation,
            dropout=hyperparameters.get("dropout", 0.0),
            basis=basis,
        )

    @staticmethod
    def hyperparameter_search_space() -> dict[str, Any]:
        return {
            "num_layers": optuna.distributions.IntDistribution(3, 9, step=1),
            "hidden_size": optuna.distributions.IntDistribution(128, 2048, step=128),
            "activation_function": ["GELU", "SiLU"],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.basis(x))
