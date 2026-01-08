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

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Convert input tuple to single tensor
        input_tensor = torch.cat(x, dim=1)

        # Perform inference
        output = self.model(input_tensor)

        # Convert output tensor to tuple of tensors
        return torch.unbind(output, dim=1)