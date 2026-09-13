"""
GDS layer map of the IHP SG13G2 open PDK.

Layers are plain ``(layer, datatype)`` tuples so geometry code can pass them straight to
gdsfactory (``layer=SG13G2.TopMetal2``), gdspy/klayout or anything else that speaks GDS
without importing the ``ihp-gdsfactory`` package (which drags in sax/JAX and pins Python).
Numbers follow the SG13G2 layer table (``sg13g2.lyp``).

Only the layers relevant for passive RF structures (metal stack, vias, pads, markers) are
listed here. Use ``SG13G2.with_purpose`` for the remaining datatypes of a layer.
"""

from typing import Final

Layer = tuple[int, int]

# Datatypes ("purposes") shared by all SG13G2 layers
PURPOSE_DRAWING: Final = 0
PURPOSE_LABEL: Final = 1
PURPOSE_PIN: Final = 2
PURPOSE_NET: Final = 3
PURPOSE_BOUNDARY: Final = 4
PURPOSE_NOFILL: Final = 23
PURPOSE_TEXT: Final = 25


class SG13G2:
    """IHP SG13G2 layers as ``(layer, datatype)`` tuples, all with the *drawing* datatype."""

    # --- front end ---
    Activ: Final[Layer] = (1, 0)
    GatPoly: Final[Layer] = (5, 0)
    Cont: Final[Layer] = (6, 0)
    NWell: Final[Layer] = (31, 0)
    PWell: Final[Layer] = (46, 0)
    PWellBlock: Final[Layer] = (46, 21)
    Substrate: Final[Layer] = (40, 0)

    # --- thin metal stack (bottom to top) ---
    Metal1: Final[Layer] = (8, 0)
    Via1: Final[Layer] = (19, 0)
    Metal2: Final[Layer] = (10, 0)
    Via2: Final[Layer] = (29, 0)
    Metal3: Final[Layer] = (30, 0)
    Via3: Final[Layer] = (49, 0)
    Metal4: Final[Layer] = (50, 0)
    Via4: Final[Layer] = (66, 0)
    Metal5: Final[Layer] = (67, 0)

    # --- thick top metal stack ---
    TopVia1: Final[Layer] = (125, 0)
    TopMetal1: Final[Layer] = (126, 0)
    TopVia2: Final[Layer] = (133, 0)
    TopMetal2: Final[Layer] = (134, 0)

    # --- MIM capacitor ---
    MIM: Final[Layer] = (36, 0)
    Vmim: Final[Layer] = (129, 0)

    # --- passivation / pads / backside ---
    Passiv: Final[Layer] = (9, 0)
    dfpad: Final[Layer] = (41, 0)
    EdgeSeal: Final[Layer] = (39, 0)
    LBE: Final[Layer] = (157, 0)  # localized backside etching

    # --- markers / annotation ---
    IND: Final[Layer] = (27, 0)  # inductor recognition layer
    NoRCX: Final[Layer] = (148, 0)  # excluded from parasitic extraction
    Recog: Final[Layer] = (99, 0)
    TEXT: Final[Layer] = (63, 0)

    # Metal layers in physical order, handy for iterating over the stack
    METAL_STACK: Final[tuple[Layer, ...]] = (
        Metal1,
        Metal2,
        Metal3,
        Metal4,
        Metal5,
        TopMetal1,
        TopMetal2,
    )

    @staticmethod
    def with_purpose(layer: Layer, purpose: int) -> Layer:
        """
        Returns the same GDS layer with a different datatype, e.g.
        ``SG13G2.with_purpose(SG13G2.TopMetal2, PURPOSE_PIN)`` -> ``(134, 2)``.
        """
        return (layer[0], purpose)
