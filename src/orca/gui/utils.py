import importlib.util
import sys
import inspect
from pathlib import Path
from typing import List, Type, Any

from orca.geometry.base_geometry import BaseGeometry
from orca.pipeline.pipeline_stage import PipelineStage

# Import all pipeline stages
from orca.pipeline.gds_gen_stage import GDSGenerator
from orca.pipeline.gds_conversion_stage import GDSConverter
from orca.pipeline.simulation_stage import PalaceSimulator
from orca.pipeline.training_stage import ModelTrainer
from orca.pipeline.export_onnx_stage import OnnxExporter
from orca.pipeline.test_model_stage import ModelTester

# Import all preset geometries 
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

def load_class_from_file(file_path: str, base_class: Type) -> Type[Any] | None:
    """
    Loads a class that inherits from `base_class` from a given file path.
    """
    path = Path(file_path)
    if not path.exists() or not path.suffix == ".py":
        return None

    # Module name from file name
    module_name = path.stem
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading module {file_path}: {e}")
        return None

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            # Avoid importing abstract classes or the base class itself if it's imported in the file
             if not inspect.isabstract(obj):
                return obj
    return None

def get_available_stages() -> List[Type[PipelineStage]]:
    """
    Returns a list of available PipelineStage subclasses.
    """
    return [
        GDSGenerator,
        GDSConverter,
        PalaceSimulator,
        ModelTrainer,
        OnnxExporter,
        ModelTester
    ]

def get_preset_geometries() -> List[Type[BaseGeometry]]:
    """
    Returns a list of preset BaseGeometry subclasses.
    """
    return [TransformerOcta]
