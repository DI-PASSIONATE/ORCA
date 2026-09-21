# Import stuff here so that they are available at the package level (i.e. from orca import ORCA, BaseGeometry, InputParameters)

from .geometry.base_geometry import BaseGeometry
from .geometry.input_parameters import InputParameterIterator
from .orca import ORCA
from .pipeline.context import PipelineContext
from .pipeline.drc_stage import DRCChecker
from .pipeline.gds_conversion_stage import GDSConverter
from .pipeline.gds_gen_stage import GDSGenerator
from .pipeline.pipeline_stage import PipelineStage
from .pipeline.simulation_stage import PalaceSimulator
from .training.codecs import FlatReImCodec, OutputCodec, UpperTriangleReImCodec
from .training.guarantees import PhysicsGuarantees
from .training.spec import FrequencyMode, IOSpec

__all__ = [
    "ORCA",
    "BaseGeometry",
    "DRCChecker",
    "FlatReImCodec",
    "FrequencyMode",
    "GDSConverter",
    "GDSGenerator",
    "IOSpec",
    "InputParameterIterator",
    "OutputCodec",
    "PalaceSimulator",
    "PhysicsGuarantees",
    "PipelineContext",
    "PipelineStage",
    "UpperTriangleReImCodec",
]

# Everything that needs PyTorch (the "train" extra) is imported on first access
# instead of here, so that `import orca` works in a simulation-only install, e.g.
# on an HPC cluster. The names stay importable as `orca.ModelTrainer` etc.; a
# missing torch is reported the moment such a name is used.
_TRAINING_EXPORTS = {
    "ModelTrainer": ".pipeline.training_stage",
    "OnnxExporter": ".pipeline.export_onnx_stage",
    "ModelTester": ".pipeline.test_model_stage",
    "OrcaModel": ".training.models.base_model",
    "register_model": ".training.models.base_model",
    "available_models": ".training.models.base_model",
    "OrcaMLP": ".training.models.mlp",
    "BasisExpansion": ".training.basis_expansion",
    "IdentityBasis": ".training.basis_expansion",
    "ChebyshevBasis": ".training.basis_expansion",
    "register_basis": ".training.basis_expansion",
    "available_bases": ".training.basis_expansion",
    "Trainer": ".training.trainer",
    "TrainingConfig": ".training.trainer",
    "TrainingResult": ".training.trainer",
    "HyperparameterTuner": ".training.tuner",
    "ComplexMSELoss": ".training.losses",
    "MSEPlusLogCoshLoss": ".training.losses",
    "NetworkPredictor": ".training.predictors",
    "TorchNetworkPredictor": ".training.predictors",
    "OnnxNetworkPredictor": ".training.predictors",
}

#: Top-level modules provided by the "train" extra in pyproject.toml.
_TRAIN_EXTRA_MODULES = frozenset(
    {"torch", "sklearn", "optuna", "onnx", "onnxruntime", "onnxscript"}
)


def __getattr__(name: str):
    module_name = _TRAINING_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as e:
        package = e.name.split(".")[0] if e.name else None
        if package not in _TRAIN_EXTRA_MODULES:
            raise
        raise ModuleNotFoundError(
            f"orca.{name} needs the '{package}' package, which is part of ORCA's "
            "optional training dependencies. Install them with "
            "`pip install -e '.[train]'` (or `uv sync --extra train`); "
            "GDS generation, conversion and Palace simulation work without them.",
            name=e.name,
        ) from e
    value = getattr(module, name)
    globals()[name] = value  # cache, so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(_TRAINING_EXPORTS))
