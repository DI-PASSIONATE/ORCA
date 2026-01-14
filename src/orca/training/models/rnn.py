import torch
import torch.nn as nn
from orca.training.normalize import Normalizer
from orca.training.feature_transform import FeatureTransformPipeline

class OrcaRNN(nn.Module):
    """
    Uses a Recurrent Neural Network (RNN) architecture for regression tasks.
    This allows the model to capture sequential dependencies (e.g. S-parameters over frequency)
    and thus potentially improve performance on such data.
    """
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, output_dim: int, normalizer: Normalizer, features: FeatureTransformPipeline | None = None):
        """
        Args:
            input_dim (int): Amount of input parameters.
            hidden_size (int): Size of the hidden state in the RNN.
            num_layers (int): Number of stacked RNN layers.
            output_dim (int): Amount of output parameters.
            normalizer (Normalizer): Normalizer for input data.
            features (FeatureTransformPipeline | None): Optional feature transformation pipeline.
        """
        super(OrcaRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )

        self.fc = nn.Linear(hidden_size, output_dim)
        self.normalizer = normalizer
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate additional features if any
        if self.features is not None:
            x = self.features(x)

        # Training data was normalized, so inference data must be normalized too
        if self.normalizer is not None:
            x = self.normalizer(x)

        # Pass through RNN
        rnn_out, _ = self.rnn(x)

        # Take the output from the last time step
        last_time_step = rnn_out[:, -1, :]

        # Pass through fully connected layer to get final output
        output = self.fc(last_time_step)

        return output