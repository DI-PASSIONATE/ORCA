from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import onnxruntime as ort


class InferenceEngine:
    """Handle ONNX model inference with checkpoint discovery."""
    
    def __init__(self, checkpoint_dir: str | Path = None):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_dir: Directory containing ONNX checkpoint files.
                           Defaults to src/orca/checkpoints
        """
        if checkpoint_dir is None:
            # Default to src/orca/checkpoints
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.model_path = None
        self.input_specs = {}
        self.output_specs = {}
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available ONNX checkpoints.
        
        Returns:
            List of checkpoint filenames (without paths)
        """
        onnx_files = list(self.checkpoint_dir.glob("*.onnx"))
        return sorted([f.name for f in onnx_files])
    
    def load_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Load an ONNX checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint file (e.g., 'model.onnx')
        
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        
        if not checkpoint_path.suffix.lower() == ".onnx":
            raise ValueError(
                f"File is not an ONNX model: {checkpoint_path}"
            )
        
        try:
            # Create session with CPU provider
            self.session = ort.InferenceSession(
                str(checkpoint_path),
                providers=['CPUExecutionProvider']
            )
            self.model_path = checkpoint_path
            
            # Extract input/output specifications
            self._extract_specs()
            
            return True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_name}: {str(e)}"
            )
    
    def _extract_specs(self):
        """Extract input and output specifications from the loaded model."""
        if self.session is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        self.input_specs = {}
        self.output_specs = {}
        
        # Get input specifications
        for input_spec in self.session.get_inputs():
            self.input_specs[input_spec.name] = {
                "name": input_spec.name,
                "shape": input_spec.shape,
                "type": input_spec.type,
            }
        
        # Get output specifications
        for output_spec in self.session.get_outputs():
            self.output_specs[output_spec.name] = {
                "name": output_spec.name,
                "shape": output_spec.shape,
                "type": output_spec.type,
            }
    
    def get_input_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get input specifications of the loaded model.
        
        Returns:
            Dictionary with input names as keys and specifications as values
        """
        return self.input_specs.copy()
    
    def get_output_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get output specifications of the loaded model.
        
        Returns:
            Dictionary with output names as keys and specifications as values
        """
        return self.output_specs.copy()
    
    def get_input_names(self) -> List[str]:
        """Get list of input names."""
        return list(self.input_specs.keys())
    
    def get_output_names(self) -> List[str]:
        """Get list of output names."""
        return list(self.output_specs.keys())
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on input data.
        
        Args:
            inputs: Dictionary mapping input names to numpy arrays
        
        Returns:
            Dictionary mapping output names to result arrays
        """
        if self.session is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        # Validate inputs
        for input_name in self.get_input_names():
            if input_name not in inputs:
                raise ValueError(
                    f"Missing required input: {input_name}"
                )
        
        # Run inference
        try:
            output_names = self.get_output_names()
            results = self.session.run(
                output_names,
                {k: v.astype(np.float32) for k, v in inputs.items()}
            )
            
            # Return as dictionary
            return {
                name: result
                for name, result in zip(output_names, results)
            }
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def infer_batch(
        self,
        input_batch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on batch input data.
        
        Args:
            input_batch: Dictionary mapping input names to batched numpy arrays
        
        Returns:
            Dictionary mapping output names to result arrays
        """
        return self.infer(input_batch)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded model."""
        if self.session is None:
            raise RuntimeError("No model loaded. Call load_checkpoint() first.")
        
        return {
            "path": str(self.model_path),
            "inputs": self.input_specs,
            "outputs": self.output_specs,
        }
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.session is not None
