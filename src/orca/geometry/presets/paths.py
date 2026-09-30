import os
from enum import StrEnum

#: Root of the presets: one folder per device (``inductor/``, ``transformer/``) holding
#: the geometry classes next to their Palace simulation configs, and the gds2palace
#: layer stackups shared by all of them in ``stackups/``.
PRESETS_DIR = os.path.dirname(__file__)

_STACKUPS_DIR = os.path.join(PRESETS_DIR, "stackups")


class StackupXML(StrEnum):
    """Absolute paths of the gds2palace stackup XMLs shipped with the presets.

    Members are ``str``, so one can be assigned to ``BaseGeometry.stackup_xml`` directly,
    e.g. ``stackup_xml: str = StackupXML.SG13G2_FEM_200um``.
    """

    #: SG13G2, planar SiO2 + passivation over TopMetal2, 200 µm chip.
    SG13G2_FEM_200um = os.path.join(_STACKUPS_DIR, "SG13G2_FEM_200um.xml")
    #: SG13G2, conformal SiO2 around TopMetal2 (derived layers), 200 µm chip.
    SG13G2_FEM_200um_passi3D = os.path.join(_STACKUPS_DIR, "SG13G2_FEM_200um_passi3D.xml")
