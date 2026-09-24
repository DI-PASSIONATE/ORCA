"""
Design-rule checks for the IHP SG13G2 back end of line, run on GDS files with KLayout.

The rules cover what a parametric passive can get wrong: vertices off the
manufacturing grid, edge angles other than 0/45/90 degrees, acute corners,
metal width and spacing, and via size, spacing and metal enclosure. The
values are those of the PDK's KLayout rule deck (``sg13g2_tech_default.json``)
and the rule names follow its report (``TM2.a``, ``TV2.c``, ...), so a finding
here can be looked up in the IHP design rule manual directly.

Two entry points: :func:`snap_to_grid` moves every vertex of a layout onto
the grid, and :func:`check_layout` counts the violations per rule. Only the
*drawing* datatype of each layer is checked, as in the PDK deck; pin, text and
the gds2palace port layers are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import klayout.db as kdb

from orca.geometry.layers import SG13G2

if TYPE_CHECKING:
    from orca.geometry.layers import Layer

#: SG13G2 manufacturing grid in nanometres (rule G.a; the PDK deck checks 5 nm).
GRID_NM: Final = 5

#: Corners with an interior angle below this are not allowed (rule 3.2, "Acute").
ACUTE_LIMIT_DEG: Final = 87.0


@dataclass(frozen=True)
class MetalRules:
    """Width and spacing rules of one metal layer, in micrometres."""

    name: str
    """Rule prefix used in the PDK deck, e.g. ``"TM2"`` for TopMetal2."""
    layer: Layer
    min_width: float
    """Rule ``<name>.a``: minimum width."""
    min_space: float
    """Rule ``<name>.b``: minimum space or notch."""


@dataclass(frozen=True)
class ViaRules:
    """Size, spacing and enclosure rules of one via layer, in micrometres."""

    name: str
    """Rule prefix used in the PDK deck, e.g. ``"TV2"`` for TopVia2."""
    layer: Layer
    size: float
    """Rule ``<name>.a``: vias are squares of exactly this size."""
    min_space: float
    """Rule ``<name>.b``: minimum space between vias."""
    enclosures: tuple[tuple[str, Layer, float], ...] = ()
    """``(rule suffix, metal layer, minimum enclosure)`` for the metals above and below."""


#: Metal rules of the SG13G2 stack: Metal1 (M1), Metal2-5 (Mn), TopMetal1/2 (TM1/TM2).
SG13G2_METAL_RULES: Final[tuple[MetalRules, ...]] = (
    MetalRules("M1", SG13G2.Metal1, min_width=0.16, min_space=0.18),
    MetalRules("M2", SG13G2.Metal2, min_width=0.20, min_space=0.21),
    MetalRules("M3", SG13G2.Metal3, min_width=0.20, min_space=0.21),
    MetalRules("M4", SG13G2.Metal4, min_width=0.20, min_space=0.21),
    MetalRules("M5", SG13G2.Metal5, min_width=0.20, min_space=0.21),
    MetalRules("TM1", SG13G2.TopMetal1, min_width=1.64, min_space=1.64),
    MetalRules("TM2", SG13G2.TopMetal2, min_width=2.00, min_space=2.00),
)

#: Via rules of the SG13G2 stack: Via1 (V1), Via2-4 (Vn), TopVia1/2 (TV1/TV2).
SG13G2_VIA_RULES: Final[tuple[ViaRules, ...]] = (
    ViaRules("V1", SG13G2.Via1, size=0.19, min_space=0.22, enclosures=(("c", SG13G2.Metal1, 0.01),)),
    ViaRules("V2", SG13G2.Via2, size=0.19, min_space=0.22, enclosures=(("c", SG13G2.Metal2, 0.005),)),
    ViaRules("V3", SG13G2.Via3, size=0.19, min_space=0.22, enclosures=(("c", SG13G2.Metal3, 0.005),)),
    ViaRules("V4", SG13G2.Via4, size=0.19, min_space=0.22, enclosures=(("c", SG13G2.Metal4, 0.005),)),
    ViaRules(
        "TV1",
        SG13G2.TopVia1,
        size=0.42,
        min_space=0.42,
        enclosures=(("c", SG13G2.Metal5, 0.10), ("d", SG13G2.TopMetal1, 0.42)),
    ),
    ViaRules(
        "TV2",
        SG13G2.TopVia2,
        size=0.90,
        min_space=1.06,
        enclosures=(("c", SG13G2.TopMetal1, 0.50), ("d", SG13G2.TopMetal2, 0.50)),
    ),
)

#: SG13G2 layer names by ``(layer, datatype)``, for the grid/angle rule names.
_LAYER_NAMES: Final[dict[Layer, str]] = {
    value: name
    for name, value in vars(SG13G2).items()
    if isinstance(value, tuple) and len(value) == 2 and not name.startswith("_")
}


@dataclass
class DRCResult:
    """Outcome of checking one layout."""

    snapped_vertices: int = 0
    """Vertices moved onto the grid before checking (0 when snapping was off)."""
    violations: dict[str, int] = field(default_factory=dict)
    """Violation count per rule name; rules without findings are absent."""

    @property
    def clean(self) -> bool:
        return not self.violations

    @property
    def total(self) -> int:
        return sum(self.violations.values())


def _snap(value: int, grid: int) -> int:
    # Python's round-half-to-even keeps a layout mirrored about the origin
    # symmetric after snapping; floor(x + 0.5) would not.
    return round(value / grid) * grid


def snap_to_grid(layout: kdb.Layout, grid_nm: int = GRID_NM) -> int:
    """Snap every vertex of every cell in *layout* onto the manufacturing grid, in place.

    Paths and polygons from ``gf.Path.extrude()`` or ``gdspy.FlexPath`` have
    perpendicular offsets on angled segments that land off the grid; this fixes
    them without flattening, so cell references and their offsets stay as they are.
    Paths with a width are turned into polygons first (their outline is what the
    mask sees); zero-width paths, which gds2palace reads as port sheets, keep
    their path form. Text labels are untouched.

    Args:
        layout: Layout to modify; its database unit must divide the grid.
        grid_nm: Grid pitch in nanometres.

    Returns:
        int: Number of vertices that were moved.
    """
    grid = _grid_dbu(layout, grid_nm)
    moved = 0

    def snap_point(p: kdb.Point) -> kdb.Point:
        nonlocal moved
        q = kdb.Point(_snap(p.x, grid), _snap(p.y, grid))
        if q != p:
            moved += 1
        return q

    for cell in layout.each_cell():
        for layer_index in layout.layer_indexes():
            for shape in cell.shapes(layer_index).each():
                if shape.is_text():
                    continue
                if shape.is_box():
                    box = shape.box
                    shape.box = kdb.Box(snap_point(box.p1), snap_point(box.p2))
                elif shape.is_path() and shape.path.width == 0:
                    path = shape.path
                    shape.path = kdb.Path([snap_point(p) for p in path.each_point()], 0)
                else:
                    # polygons, simple polygons and paths with a width
                    polygon = shape.polygon
                    snapped = kdb.Polygon([snap_point(p) for p in polygon.each_point_hull()])
                    for hole in range(polygon.holes()):
                        snapped.insert_hole([snap_point(p) for p in polygon.each_point_hole(hole)])
                    shape.polygon = snapped
    return moved


def check_layout(
    layout: kdb.Layout,
    grid_nm: int = GRID_NM,
    metal_rules: tuple[MetalRules, ...] = SG13G2_METAL_RULES,
    via_rules: tuple[ViaRules, ...] = SG13G2_VIA_RULES,
) -> dict[str, int]:
    """Count the design-rule violations of *layout*, per rule.

    Every layer in the rule tables is flattened into one region (merged, as the
    PDK deck does) and checked for:

    - ``<layer>.offgrid``: vertices off the ``grid_nm`` grid;
    - ``<layer>.angle``: edges that are not 0/45/90 degrees on metals, or not
      0/90 degrees on vias;
    - ``<layer>.acute``: corners sharper than :data:`ACUTE_LIMIT_DEG`;
    - ``<rule>.a``/``.b``: minimum width and space of metals, exact size and
      minimum space of vias;
    - ``<rule>.c``/``.d``: metal enclosure of vias. A via not covered by the
      metal at all counts here too, which the edge-based PDK check would miss.

    Args:
        layout: Layout to check. It is not modified.
        grid_nm: Manufacturing grid in nanometres.
        metal_rules: Metal layers to check; defaults to the SG13G2 stack.
        via_rules: Via layers to check; defaults to the SG13G2 stack.

    Returns:
        dict[str, int]: Violation counts keyed by rule name; only rules with
        at least one finding are present, so an empty dict means clean.
    """
    grid = _grid_dbu(layout, grid_nm)
    dbu = layout.dbu
    to_dbu = lambda um: round(um / dbu)  # noqa: E731 - one-line unit helper
    regions: dict[Layer, kdb.Region] = {}

    def region(layer: Layer) -> kdb.Region:
        if layer not in regions:
            index = layout.find_layer(*layer)
            merged = kdb.Region()
            if index is not None:
                for top in layout.top_cells():
                    merged += kdb.Region(top.begin_shapes_rec(index))
            regions[layer] = merged.merged()
        return regions[layer]

    violations: dict[str, int] = {}

    def record(rule: str, count: int) -> None:
        if count:
            violations[rule] = violations.get(rule, 0) + count

    def geometry_checks(layer: Layer, diagonal: bool) -> None:
        polygons = region(layer)
        if polygons.is_empty():
            return
        name = _LAYER_NAMES.get(layer, f"{layer[0]}/{layer[1]}")
        record(f"{name}.offgrid", polygons.grid_check(grid, grid).count())
        edges = polygons.edges().with_abs_angle(0, True).with_abs_angle(90, True)
        if diagonal:
            edges = edges.with_abs_angle(45, True)
        record(f"{name}.angle", edges.count())
        # KLayout reports the turning angle at a corner; an interior angle below
        # the limit is a turn sharper than 180 - limit. 'absolute' folds concave
        # corners in, so sharp notches are caught as well.
        acute = polygons.corners(180.0 - ACUTE_LIMIT_DEG, 180.0, 2, False, True, False, True)
        record(f"{name}.acute", acute.count())

    for metal in metal_rules:
        polygons = region(metal.layer)
        if polygons.is_empty():
            continue
        geometry_checks(metal.layer, diagonal=True)
        record(f"{metal.name}.a", _check_count(polygons.width_check, to_dbu(metal.min_width)))
        record(f"{metal.name}.b", _check_count(polygons.space_check, to_dbu(metal.min_space)))

    for via in via_rules:
        vias = region(via.layer)
        if vias.is_empty():
            continue
        geometry_checks(via.layer, diagonal=False)
        size = to_dbu(via.size)
        wrong_min = vias.with_bbox_min(size, size + 1, True)
        wrong_max = vias.with_bbox_min(size, size + 1, False).with_bbox_max(size, size + 1, True)
        record(f"{via.name}.a", wrong_min.count() + wrong_max.count())
        record(f"{via.name}.b", _check_count(vias.space_check, to_dbu(via.min_space)))
        for suffix, metal_layer, enclosure in via.enclosures:
            metal = region(metal_layer)
            uncovered = (vias - metal).count()
            too_close = vias.enclosed_check(
                metal, to_dbu(enclosure), False, kdb.Region.Euclidian
            ).count()
            record(f"{via.name}.{suffix}", uncovered + too_close)

    return violations


def check_gds_file(path: str, grid_nm: int = GRID_NM, snap: bool = True) -> DRCResult:
    """Check one GDS file, optionally snapping it to the grid first.

    Args:
        path: GDS file to check.
        grid_nm: Manufacturing grid in nanometres.
        snap: Snap all vertices to the grid and write the file back before
            checking. Off-grid vertices are then repaired rather than reported.

    Returns:
        DRCResult: Vertices moved and the violations found.
    """
    layout = kdb.Layout()
    layout.read(path)
    moved = 0
    if snap:
        moved = snap_to_grid(layout, grid_nm)
        if moved:
            layout.write(path)
    return DRCResult(snapped_vertices=moved, violations=check_layout(layout, grid_nm))


def _grid_dbu(layout: kdb.Layout, grid_nm: int) -> int:
    """The grid in database units, checking that the layout can represent it."""
    if grid_nm <= 0:
        raise ValueError(f"grid_nm must be positive, got {grid_nm}.")
    grid = grid_nm * 1e-3 / layout.dbu
    if abs(grid - round(grid)) > 1e-6 or round(grid) < 1:
        raise ValueError(
            f"A {grid_nm} nm grid is not a multiple of the layout's database unit "
            f"({layout.dbu * 1e3:g} nm)."
        )
    return round(grid)


def _check_count(check, distance: int) -> int:
    """Number of edge pairs a KLayout width/space check reports for *distance* (Euclidean)."""
    return check(distance, False, kdb.Region.Euclidian).count()
