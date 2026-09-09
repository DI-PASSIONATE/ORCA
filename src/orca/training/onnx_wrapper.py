import torch
from orca.training.normalize import Normalizer


class ONNXWrapper(torch.nn.Module):
    """
    A wrapper for ONNX models to adapt input/output formats.
    This class allows an ONNX model to be exported with multiple inputs and outputs,
    while internally handling a single concatenated input and output tensor.
    Therefore, you can export an ONNX model that takes multiple inputs and produces multiple outputs and name
    them accordingly.

    Args:
        model: The PyTorch model to be wrapped.
    Returns:
        A torch.nn.Module that can be passed to torch.onnx.export.
    """

    def __init__(
        self,
        model,
        input_normalizer: Normalizer,
        output_denormalizer: Normalizer,
    ):
        super().__init__()
        self.model = model
        self.input_normalizer = input_normalizer
        self.output_denormalizer = output_denormalizer

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Convert input tuple to single tensor
        input_tensor = torch.cat(x, dim=1)

        # Normalizing the input
        if self.input_normalizer is not None:
            input_tensor = self.input_normalizer(input_tensor)

        # Perform inference. Any basis expansion lives inside the model, so it is
        # traced into the exported graph here rather than applied separately.
        output = self.model(input_tensor)

        # De-normalize output
        if self.output_denormalizer is not None:
            output = self.output_denormalizer.denormalize(output)

        # Convert output tensor to tuple of tensors
        return torch.unbind(output, dim=1)
