from typing_extensions import Self
import torch.nn as nn

class OrcaMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(OrcaMLP, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)