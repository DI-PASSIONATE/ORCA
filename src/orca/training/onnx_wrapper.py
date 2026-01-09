import torch

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

    def __init__(self, model, output_means: list[float], output_stds: list[float]):
        super().__init__()
        self.model = model
        # We register means and stds as buffers to ensure they are part of the model state
        self.register_buffer("output_means", torch.tensor(output_means, dtype=torch.float32))
        self.register_buffer("output_stds", torch.tensor(output_stds, dtype=torch.float32))

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Convert input tuple to single tensor
        input_tensor = torch.cat(x, dim=1)

        # Perform inference
        output = self.model(input_tensor)

        # De-normalize output
        output = output * self.output_stds + self.output_means

        # Convert output tensor to tuple of tensors
        return torch.unbind(output, dim=1)