"""Shape contract between datasets and models.

A dataset knows what it produces; a model declares what it consumes. :class:`IOSpec`
is the handshake between them, so no model has to hardcode input or output
dimensions that the dataset already knows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from orca.training.codecs import OutputCodec


class FrequencyMode(Enum):
    """How frequency is laid out in a sample.

    Attributes:
        PER_POINT: One sample per frequency point; frequency is an input column.
        BAND: One sample per geometry; the target covers the whole frequency grid.
        CONTINUOUS: The model is evaluated at arbitrary frequencies (e.g. an
            implicit-neural-representation or rational/pole-residue head).
    """

    PER_POINT = auto()
    BAND = auto()
    CONTINUOUS = auto()


@dataclass(frozen=True, eq=False)
class IOSpec:
    """Everything a model needs in order to size itself.

    Args:
        input_names (tuple[str, ...]): Model-facing input names, in column order.
            These become the ONNX input names.
        codec (OutputCodec): Output representation the dataset encoded its targets with.
        frequency_mode (FrequencyMode): Frequency layout of the dataset samples.
        n_extra_features (int): Number of extra columns appended by the feature
            transform pipeline. Not part of ``input_names``, since they are
            computed inside the exported model rather than passed in.
        frequency_grid (np.ndarray | None): Frequency points of a BAND dataset.
    """

    input_names: tuple[str, ...]
    codec: OutputCodec
    frequency_mode: FrequencyMode
    n_extra_features: int = 0
    frequency_grid: np.ndarray | None = None

    @property
    def input_dim(self) -> int:
        """Width of the tensor the model actually sees, features included."""
        return len(self.input_names) + self.n_extra_features

    @property
    def output_dim(self) -> int:
        return self.codec.output_dim

    @property
    def output_names(self) -> list[str]:
        return self.codec.output_names

    @property
    def n_ports(self) -> int:
        return self.codec.n_ports
