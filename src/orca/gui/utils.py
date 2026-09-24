import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from orca.geometry.base_geometry import BaseGeometry
from orca.geometry.presets.inductor_octa import InductorOcta
from orca.logger import logger
from orca.pipeline.drc_stage import DRCChecker
from orca.pipeline.gds_conversion_stage import GDSConverter

# Import all pipeline stages
from orca.pipeline.gds_gen_stage import GDSGenerator
from orca.pipeline.pipeline_stage import PipelineStage
from orca.pipeline.simulation_stage import PalaceSimulator

# The training stages need PyTorch (the "train" extra). Without it the GUI still
# offers the GDS generation, conversion and simulation stages.
try:
    from orca.pipeline.export_onnx_stage import OnnxExporter
    from orca.pipeline.test_model_stage import ModelTester
    from orca.pipeline.training_stage import ModelTrainer
except ModuleNotFoundError as e:
    _TRAINING_STAGES: list[type[PipelineStage]] = []
    logger.warning(
        f"Training stages are unavailable ({e}). Install ORCA's optional "
        "training dependencies with `pip install -e '.[train]'` to enable them."
    )
else:
    _TRAINING_STAGES = [ModelTrainer, OnnxExporter, ModelTester]

# Import all preset geometries
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta


def load_class_from_file(file_path: str, base_class: type) -> type[Any] | None:
    """
    Loads a class that inherits from `base_class` from a given file path.
    """
    path = Path(file_path)
    if not path.exists() or path.suffix != ".py":
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
    except Exception as e:  # noqa: BLE001 - a broken user file must not take the GUI down
        logger.error(f"Error loading module {file_path}: {e}")
        return None

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        # Avoid importing abstract classes or the base class itself if it's imported in the file
        if issubclass(obj, base_class) and obj is not base_class and not inspect.isabstract(obj):
            return obj
    return None

def get_available_stages() -> list[type[PipelineStage]]:
    """
    Returns a list of available PipelineStage subclasses.
    """
    return [GDSGenerator, DRCChecker, GDSConverter, PalaceSimulator, *_TRAINING_STAGES]

def get_preset_geometries() -> list[type[BaseGeometry]]:
    """
    Returns a list of preset BaseGeometry subclasses.
    """
    return [TransformerOcta, InductorOcta]
