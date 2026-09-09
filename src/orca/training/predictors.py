"""Turning geometry parameters into a predicted network.

A :class:`NetworkPredictor` hides *how* a prediction is produced - a torch model
plus the dataset's preprocessing, or an exported ONNX session - behind a single
call that returns a ``skrf.Network``. Evaluation code can then be written once and
work for either, and for any output representation, since decoding is delegated to
the model's :class:`~orca.training.codecs.OutputCodec`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import skrf as rf
import torch

from orca.training.codecs import OutputCodec
from orca.training.models.base_model import OrcaModel
from orca.training.spec import FrequencyMode


class NetworkPredictor(ABC):
    """Predicts the response of a geometry over a frequency grid."""

    @abstractmethod
    def predict(self, params: np.ndarray, frequencies: np.ndarray) -> rf.Network:
        """
        Args:
            params (np.ndarray): Geometry parameters, in the order the model was trained on.
            frequencies (np.ndarray): Frequency points in Hz.

        Returns:
            rf.Network: Predicted network sampled at ``frequencies``.
        """


class TorchNetworkPredictor(NetworkPredictor):
    """Runs a trained torch model, reusing the dataset's preprocessing.

    The feature pipeline and normalizers live on the dataset, so the same
    transformations that were applied during training are applied here. This is
    what the exported ONNX model does internally via
    :class:`~orca.training.onnx_wrapper.ONNXWrapper`.

    Args:
        model (OrcaModel): Trained model.
        dataset: Dataset the model was trained on, supplying features, normalizers and codec.
    """

    def __init__(self, model: OrcaModel, dataset):
        if model.frequency_mode is not FrequencyMode.PER_POINT:
            raise NotImplementedError(
                f"{type(model).__name__} consumes {model.frequency_mode.name} samples; "
                "TorchNetworkPredictor currently only builds inputs for PER_POINT models."
            )
        self.model = model
        self.dataset = dataset
        self.device = dataset.device

    def predict(self, params: np.ndarray, frequencies: np.ndarray) -> rf.Network:
        self.model.to(self.device).eval()

        # One row per frequency point: geometry parameters repeated, frequency appended
        x = np.hstack(
            [
                np.repeat(np.asarray(params, dtype=np.float32)[None, :], len(frequencies), axis=0),
                np.asarray(frequencies, dtype=np.float32)[:, None],
            ]
        ).astype(np.float32)
        x = torch.tensor(x, device=self.device)

        if self.dataset.features is not None:
            x = self.dataset.features(x)
        if self.dataset.input_normalizer is not None:
            x = self.dataset.input_normalizer.normalize(x)

        with torch.no_grad():
            raw = self.model(x)

        if self.dataset.output_normalizer is not None:
            raw = self.dataset.output_normalizer.denormalize(raw)

        return self.model.to_network(raw.cpu().numpy(), frequencies)


class OnnxNetworkPredictor(NetworkPredictor):
    """Runs an exported ONNX model through onnxruntime.

    The exported graph takes one named scalar column per input and emits one named
    scalar column per output value, so inputs are broadcast over the frequency grid
    and outputs are stacked back into the codec's layout.

    Args:
        session (onnxruntime.InferenceSession): Session for the exported model.
        codec (OutputCodec): Codec matching the one the model was trained with.
    """

    def __init__(self, session, codec: OutputCodec):
        self.session = session
        self.codec = codec
        self.input_names = [node.name for node in session.get_inputs()]
        self.output_names = codec.output_names

    def predict(self, params: np.ndarray, frequencies: np.ndarray) -> rf.Network:
        frequencies = np.asarray(frequencies)
        params = iter(np.asarray(params, dtype=np.float32))

        feed = {}
        for name in self.input_names:
            if name == "frequency":
                column = frequencies
            else:
                column = np.full(len(frequencies), next(params))
            feed[name] = column.reshape(-1, 1).astype(np.float32)

        outputs = self.session.run(self.output_names, feed)
        raw = np.column_stack([np.asarray(o).reshape(-1) for o in outputs])
        return self.codec.to_network(raw, frequencies)
