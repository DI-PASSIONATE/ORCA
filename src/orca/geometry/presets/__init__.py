# Re-export the preset geometries, so they import as `from orca.geometry.presets import InductorOcta`.

from .inductor.inductor_octa import InductorOcta
from .paths import PRESETS_DIR, StackupXML
from .transformer.tf_octa_c_ports import TransformerOcta

__all__ = ["PRESETS_DIR", "InductorOcta", "StackupXML", "TransformerOcta"]
