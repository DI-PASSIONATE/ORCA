import math

import gdsfactory as gf
import klayout.db as kdb
import numpy as np

from orca.geometry.drc import GRID_NM
from orca.geometry.layers import SG13G2

#: Manufacturing grid in µm.
_GRID = GRID_NM / 1000.0


def _ensure_active_pdk() -> None:
    """Gdsfactory refuses to build components without an active PDK. We only use it as
    a polygon generator with explicit SG13G2 layer tuples, so its generic PDK is enough.
    """
    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

def check_tf_octa_c_parameters(
    bottom_winding_diameter: float = 50.0,
    top_winding_diameter: float = 50.0,
    center_displacement: float = 15.0,
    bottom_linewidth: float = 5.0,
    bottom_center_tap_width: float = 0.0,
    lower_feed_type: int = 1,
    top_linewidth: float = 5.0,
    upper_center_tap_width: float = 0.0,
    upper_feed_type: int = 1,
    feedline_spacing: float = 6.0,
    gnd_upper_spacing: float = 20.0,
    gnd_lower_spacing: float = 20.0,
    gnd_side_spacing: float = 20.0,
    gnd_ring_width: float = 10.0,
) -> None:
    """Raise ``ValueError`` if :func:`tf_octa_c` cannot draw these parameters.

    Same arguments as :func:`tf_octa_c`. Called by it, and by
    ``TransformerOcta.is_feasible`` to reject a draw before anything is drawn. Only
    drawability is checked here; which combinations make a useful transformer (how
    well the windings couple) is the preset's choice.
    """
    bottom_centertap_width = (
        bottom_center_tap_width if bottom_center_tap_width > 0.1 else bottom_linewidth
    )
    top_centertap_width = (
        upper_center_tap_width if upper_center_tap_width > 0.1 else top_linewidth
    )
    for label, feed_type in (("lower_feed_type", lower_feed_type), ("upper_feed_type", upper_feed_type)):
        if feed_type not in (0, 1):
            raise NotImplementedError(
                f"{label}={feed_type}: only 0 (no center tap) and 1 (center tap) are implemented."
            )
    draw_bottom_tap = lower_feed_type == 1
    draw_top_tap = upper_feed_type == 1
    # The center tap of one winding crosses the feed gap of the other, which widens that gap.
    fs_top = max(feedline_spacing, bottom_centertap_width) if draw_bottom_tap else feedline_spacing
    fs_bot = max(feedline_spacing, top_centertap_width) if draw_top_tap else feedline_spacing

    # Check if linewidth is too large for winding diameter
    if bottom_linewidth > bottom_winding_diameter / 3.0:
        raise ValueError("bottom_linewidth is too large for input_winding_diameter.")
    if top_linewidth > top_winding_diameter / 3.0:
        raise ValueError("upper_linewidth is too large for output_winding_diameter.")
    # Check if center tap width is too large for winding diameter of the other winding
    if draw_bottom_tap and bottom_centertap_width > top_winding_diameter / 3.0:
        raise ValueError(
            "bottom_center_tap_width is too large for output_winding_diameter."
        )
    if draw_top_tap and top_centertap_width > bottom_winding_diameter / 3.0:
        raise ValueError(
            "upper_center_tap_width is too large for input_winding_diameter."
        )

    # The feed gap is cut out of the octagon's flat side facing the ports. It must fit
    # in that side along the trace's centre line (diameter * sin 22.5 deg long), so
    # that the trace ends on either side of the gap still run along the flat side.
    for label, diameter, gap in (
        ("top", top_winding_diameter, fs_top),
        ("bottom", bottom_winding_diameter, fs_bot),
    ):
        if gap > diameter * np.sin(np.radians(22.5)):
            raise ValueError(
                f"The {label} winding's feed gap of {gap:g} does not fit in the flat side of "
                f"a {diameter:g} octagon."
            )

    # The ports sit on the ring's inner edge, gnd_*_spacing - gnd_ring_width beyond the
    # windings' vertices; closer than half a trace width, a feed would run back into its
    # own winding instead of out of it.
    for label, spacing in (("gnd_upper_spacing", gnd_upper_spacing), ("gnd_lower_spacing", gnd_lower_spacing)):
        if spacing - gnd_ring_width < max(top_linewidth, bottom_linewidth) / 2.0:
            raise ValueError(
                f"{label} - gnd_ring_width = {spacing - gnd_ring_width:g} puts the ports inside "
                "the windings; it must be at least half the widest trace."
            )

    # Ground ring: the ports on both sides and the ring bars must leave a positive opening.
    tf_y = max(top_winding_diameter, bottom_winding_diameter) / 2.0 + gnd_side_spacing
    port_xr = (
        max(
            center_displacement / 2.0 + top_winding_diameter / 2.0,
            -center_displacement / 2.0 + bottom_winding_diameter / 2.0,
        )
        + gnd_upper_spacing
    )
    port_xl = (
        min(
            center_displacement / 2.0 - top_winding_diameter / 2.0,
            -center_displacement / 2.0 - bottom_winding_diameter / 2.0,
        )
        - gnd_lower_spacing
    )
    if not (port_xr - gnd_ring_width > port_xl + gnd_ring_width and tf_y - gnd_ring_width > 0):
        raise ValueError(
            "Ground ring dimensions are invalid due to port spacing. Adjust parameters."
        )


def tf_octa_c(
    name: str = "tf_octa_c",
    bottom_winding_diameter: float = 50.0,
    top_winding_diameter: float = 50.0,
    center_displacement: float = 15.0,
    bottom_linewidth: float = 5.0,
    bottom_center_tap_width: float = 0.0,
    lower_feed_type: int = 1,
    top_linewidth: float = 5.0,
    upper_center_tap_width: float = 0.0,
    upper_feed_type: int = 1,
    feedline_spacing: float = 6.0,
    gnd_upper_spacing: float = 20.0,
    gnd_lower_spacing: float = 20.0,
    gnd_side_spacing: float = 20.0,
    gnd_ring_width: float = 10.0,
    textlabel: str = "",
) -> gf.Component:
    """
    Octagon Transformer Component with Feed Extensions and Overlap Checks.

    Args:
        name: Name of the generated cell.
        bottom_winding_diameter: Diameter of the input (lower/bot) winding (trace center to center).
        top_winding_diameter: Diameter of the output (upper/top) winding (trace center to center).
        center_displacement: Displacement (offset between winding centers).
        bottom_linewidth: Trace width of the lower winding.
        bottom_center_tap_width: Trace width of the lower center tap (<= 0.1 uses bottom_linewidth).
        lower_feed_type: Center tap of the lower winding: 0 = none, 1 = tap from the back of
            the winding to the right ring edge (port ``ico`` on layer 206). 2 (tap through
            the winding's own feed gap) and 3 (both) are not implemented.
        top_linewidth: Trace width of the upper winding.
        upper_center_tap_width: Trace width of the upper center tap (<= 0.1 uses top_linewidth).
        upper_feed_type: Center tap of the upper winding, same encoding as lower_feed_type
            (port ``oci`` on layer 205 when set to 1).
        feedline_spacing: Feedline spacing (gap between inner sides of feed lines).
        gnd_upper_spacing: Ring spacing on the upper winding side.
        gnd_lower_spacing: Ring spacing on the lower winding side.
        gnd_side_spacing: Ring spacing at the side.
        gnd_ring_width: Ring width.
        textlabel: Text placed at the transformer's centre on the TEXT layer, to read the
            dimensions in a layout viewer. Empty lists every dimension drawn.
    """
    _ensure_active_pdk()
    check_tf_octa_c_parameters(
        bottom_winding_diameter=bottom_winding_diameter,
        top_winding_diameter=top_winding_diameter,
        center_displacement=center_displacement,
        bottom_linewidth=bottom_linewidth,
        bottom_center_tap_width=bottom_center_tap_width,
        lower_feed_type=lower_feed_type,
        top_linewidth=top_linewidth,
        upper_center_tap_width=upper_center_tap_width,
        upper_feed_type=upper_feed_type,
        feedline_spacing=feedline_spacing,
        gnd_upper_spacing=gnd_upper_spacing,
        gnd_lower_spacing=gnd_lower_spacing,
        gnd_side_spacing=gnd_side_spacing,
        gnd_ring_width=gnd_ring_width,
    )

    LAYER_BOT = SG13G2.TopMetal1
    LAYER_TOP = SG13G2.TopMetal2
    LAYER_RING = SG13G2.Metal5

    c = gf.Component(name)
    # -------------------------------------------------
    # 1. Variable Calculation & Overlap Check
    # -------------------------------------------------
    bottom_centertap_width = (
        bottom_center_tap_width if bottom_center_tap_width > 0.1 else bottom_linewidth
    )
    top_centertap_width = (
        upper_center_tap_width if upper_center_tap_width > 0.1 else top_linewidth
    )
    draw_bottom_tap = lower_feed_type == 1
    draw_top_tap = upper_feed_type == 1

    # --- Safety Check: Octagon Opening Width ---
    # Top Winding Gap is on the RIGHT.
    # Bot Center Tap goes RIGHT. It crosses Top Gap.
    fs_top = max(feedline_spacing, bottom_centertap_width) if draw_bottom_tap else feedline_spacing

    # Bot Winding Gap is on the LEFT.
    # Top Center Tap goes LEFT. It crosses Bot Gap.
    fs_bot = max(feedline_spacing, top_centertap_width) if draw_top_tap else feedline_spacing

    # Geometry Limits
    tf_y = max(top_winding_diameter, bottom_winding_diameter) / 2.0 + gnd_side_spacing

    # The windings are built on the manufacturing grid, so their centres are placed on it
    # too (at most 2.5 nm from the requested offset): an off-grid shift would take every
    # vertex off the grid, and snapping them afterwards could tilt the 45 degree sides.
    half_offset = round(center_displacement / 2.0 / _GRID) * _GRID

    def on_grid(x: float) -> float:
        return round(x / _GRID) * _GRID

    # X Limits for Ports, on the grid like the windings whose feeds end there
    # Note: Winding edges are approx at center +/- diameter/2
    top_right_x = half_offset + (top_winding_diameter / 2.0)
    bot_right_x = -half_offset + (bottom_winding_diameter / 2.0)
    port_xr = on_grid(max(top_right_x, bot_right_x) + gnd_upper_spacing)

    top_left_x = half_offset - (top_winding_diameter / 2.0)
    bot_left_x = -half_offset - (bottom_winding_diameter / 2.0)
    port_xl = on_grid(min(top_left_x, bot_left_x) - gnd_lower_spacing)

    # -------------------------------------------------
    # 2. Helper: Winding Generator
    # -------------------------------------------------
    def create_octa_winding(
        diameter,
        width,
        gap_size,
        layer,
        center_x,
        center_y,
        rotation_deg,
        feed_target_x,
        centertap_target_x,
        centertap_width,
    ):
        """
        Draws one winding with its feed lines and optional center tap as one polygon.

        The winding is the ring between two octagons whose flat sides lie ``width / 2``
        outside and inside the trace's centre line, with the feed gap cut out of the
        flat side facing the feeds. Unlike a mitred path, the cut never folds the
        trace, however short the side next to the gap.

        Args:
            diameter: Diameter of the trace's centre line, vertex to vertex.
            width: Trace width of the winding and its feed lines.
            gap_size: Spacing between the inner edges of the feed lines.
            layer: Metal layer of the winding.
            center_x: x of the winding's centre.
            center_y: y of the winding's centre.
            rotation_deg: 0 for feeds towards +x, 180 for feeds towards -x.
            feed_target_x: x of the feed ports.
            centertap_target_x: x of the center tap port, or None for no center tap.
            centertap_width: Trace width of the center tap.
        """
        if rotation_deg not in (0, 180):
            raise ValueError(f"rotation_deg must be 0 or 180, got {rotation_deg}.")
        dbu = c.layout().dbu
        # Built in the winding's own frame (feeds towards +x), then turned into place
        to_global = kdb.DCplxTrans(1.0, rotation_deg, False, center_x, center_y)
        direction = 1.0 if rotation_deg == 0 else -1.0

        def local_x(global_x):
            return (global_x - center_x) * direction

        def octagon(flat, offset):
            # Flat sides at +-flat facing +-x and +-y; vertices at (+-flat, +-offset) and
            # (+-offset, +-flat), so every diagonal side is exactly 45 degrees
            corners = [(flat, offset), (offset, flat), (-offset, flat), (-flat, offset)]
            corners += [(-x, -y) for x, y in corners]
            return kdb.DPolygon([kdb.DPoint(x, y) for x, y in corners])

        def region(shape):
            return kdb.Region(shape.transformed(to_global).to_itype(dbu))

        # The octagons are built on the manufacturing grid rather than snapped later:
        # snapping each vertex on its own would tilt the 45 degree sides off 45 degrees,
        # which the SG13G2 angle rule rejects. Rounding the outer octagon's vertex offset
        # up and the inner one's down keeps the diagonal trace at least `width` wide.
        tan_22 = math.tan(math.radians(22.5))
        outer = round((diameter / 2.0 * math.cos(math.radians(22.5)) + width / 2.0) / _GRID) * _GRID
        inner = outer - width
        apothem = outer - width / 2.0  # flat side of the trace's centre line
        winding = region(octagon(outer, math.ceil(outer * tan_22 / _GRID) * _GRID)) - region(
            octagon(inner, math.floor(inner * tan_22 / _GRID) * _GRID)
        )
        winding -= region(
            kdb.DPolygon(
                kdb.DBox(apothem - width, -gap_size / 2.0, apothem + width, gap_size / 2.0)
            )
        )

        # Feed lines from the trace ends straight out to the ports
        feed_end = local_x(feed_target_x)
        for y0, y1 in ((gap_size / 2.0, gap_size / 2.0 + width), (-gap_size / 2.0 - width, -gap_size / 2.0)):
            winding += region(kdb.DPolygon(kdb.DBox(apothem - width / 2.0, y0, feed_end, y1)))

        # Center tap: from the back of the winding straight out to its port
        if centertap_target_x is not None:
            winding += region(
                kdb.DPolygon(
                    kdb.DBox(
                        local_x(centertap_target_x),
                        -centertap_width / 2.0,
                        -apothem + width / 2.0,
                        centertap_width / 2.0,
                    )
                )
            )

        c.shapes(c.layout().layer(*layer)).insert(winding.merged())

    # -------------------------------------------------
    # 3. Create Geometry
    # -------------------------------------------------

    # Top Winding (Rot 0, Gap Right -> connects to port_xr)
    create_octa_winding(
        diameter=top_winding_diameter,
        width=top_linewidth,
        gap_size=fs_top,
        layer=LAYER_TOP,
        center_x=half_offset,
        center_y=0,
        rotation_deg=0,
        feed_target_x=port_xr - gnd_ring_width,
        centertap_target_x=port_xl + gnd_ring_width if draw_top_tap else None,
        centertap_width=top_centertap_width,
    )

    # Bot Winding (Rot 180, Gap Left -> connects to port_xl)
    create_octa_winding(
        diameter=bottom_winding_diameter,
        width=bottom_linewidth,
        gap_size=fs_bot,
        layer=LAYER_BOT,
        center_x=-half_offset,
        center_y=0,
        rotation_deg=180,
        feed_target_x=port_xl + gnd_ring_width,
        centertap_target_x=port_xr - gnd_ring_width if draw_bottom_tap else None,
        centertap_width=bottom_centertap_width,
    )

    # -------------------------------------------------
    # 4. Main Ports (ip, in, op, on)
    # -------------------------------------------------
    # Calculated Y centers for ports based on adjusted gap sizes
    y_top_p = fs_top / 2.0 + top_linewidth / 2.0
    y_top_n = -fs_top / 2.0 - top_linewidth / 2.0
    y_bot_p = (
        fs_bot / 2.0 + bottom_linewidth / 2.0
    )  # Bot is rotated 180, but Y logic is symmetric magnitude
    y_bot_n = -fs_bot / 2.0 - bottom_linewidth / 2.0

    # Zero-width paths on port layers create Palace's 2D vertical port sheets.

    def add_port_marker(center, width, layer, orientation):
        dbu = c.layout().dbu
        layer_index = c.layout().layer(*layer)
        angle = np.radians(orientation + 90.0)
        dx = width * np.cos(angle) / 2.0
        dy = width * np.sin(angle) / 2.0
        start = kdb.Point(
            round((center[0] - dx) / dbu), round((center[1] - dy) / dbu)
        )
        end = kdb.Point(
            round((center[0] + dx) / dbu), round((center[1] + dy) / dbu)
        )
        c.shapes(layer_index).insert(kdb.Path([start, end], 0))

    ### TOP LAYER (ports on the RIGHT) -> Port 1 and 2 -> Layer 201, 202
    # OP: top winding, right side, upper port
    c.add_port(
        name="op",
        center=(port_xr - gnd_ring_width, y_top_p),
        width=top_linewidth,
        orientation=0,
        layer=(201, 0),
    )
    add_port_marker(
        (port_xr - gnd_ring_width, y_top_p),
        top_linewidth,
        (201, 0),
        0,
    )
    # ON: top winding, right side, lower port
    c.add_port(
        name="on",
        center=(port_xr - gnd_ring_width, y_top_n),
        width=top_linewidth,
        orientation=0,
        layer=(202, 0),
    )
    add_port_marker(
        (port_xr - gnd_ring_width, y_top_n),
        top_linewidth,
        (202, 0),
        0,
    )
    # Center Tap (Top, Center) -> Layer 205
    if draw_top_tap:
        c.add_port(
            name="oci",
            center=(port_xl + gnd_ring_width, 0.0),
            width=top_centertap_width,
            orientation=180,
            layer=(205, 0),
        )
        add_port_marker(
            (port_xl + gnd_ring_width, 0.0), top_centertap_width, (205, 0), 180
        )

    ### BOT LAYER (ports on the LEFT) -> Port 3 and 4 -> Layer 203, 204
    # IP: bottom winding, left side, upper port
    c.add_port(
        name="ip",
        center=(port_xl + gnd_ring_width, y_bot_p),
        width=bottom_linewidth,
        orientation=180,
        layer=(203, 0),
    )
    add_port_marker(
        (port_xl + gnd_ring_width, y_bot_p),
        bottom_linewidth,
        (203, 0),
        180,
    )
    # IN: bottom winding, left side, lower port
    c.add_port(
        name="in",
        center=(port_xl + gnd_ring_width, y_bot_n),
        width=bottom_linewidth,
        orientation=180,
        layer=(204, 0),
    )
    add_port_marker(
        (port_xl + gnd_ring_width, y_bot_n),
        bottom_linewidth,
        (204, 0),
        180,
    )
    # Center Tap (Bot, Center) -> Layer 206
    if draw_bottom_tap:
        c.add_port(
            name="ico",
            center=(port_xr - gnd_ring_width, 0.0),
            width=bottom_centertap_width,
            orientation=0,
            layer=(206, 0),
        )
        add_port_marker(
            (port_xr - gnd_ring_width, 0.0), bottom_centertap_width, (206, 0), 0
        )

    # -------------------------------------------------
    # 6. Ground Ring
    # -------------------------------------------------
    # Ground ring as a rectangle frame
    inner_xl = port_xl + gnd_ring_width
    inner_xr = port_xr - gnd_ring_width
    inner_y = tf_y - gnd_ring_width

    if inner_xr > inner_xl and inner_y > 0:
        # Build ring from four rectangles
        outer_w = port_xr - port_xl
        outer_h = 2 * tf_y

        # Top bar
        top = gf.components.rectangle(size=(outer_w, gnd_ring_width), layer=LAYER_RING)
        top_ref = c << top
        top_ref.move((port_xl, tf_y - gnd_ring_width))

        # Bottom bar
        bot = gf.components.rectangle(size=(outer_w, gnd_ring_width), layer=LAYER_RING)
        bot_ref = c << bot
        bot_ref.move((port_xl, -tf_y))

        # Left bar
        left_h = outer_h - 2 * gnd_ring_width
        if left_h > 0:
            left = gf.components.rectangle(
                size=(gnd_ring_width, left_h), layer=LAYER_RING
            )
            left_ref = c << left
            left_ref.move((port_xl, -tf_y + gnd_ring_width))

        # Right bar
        if left_h > 0:
            right = gf.components.rectangle(
                size=(gnd_ring_width, left_h), layer=LAYER_RING
            )
            right_ref = c << right
            right_ref.move(
                (port_xr - gnd_ring_width, -tf_y + gnd_ring_width)
            )
    else:
        raise ValueError(
            "Ground ring dimensions are invalid due to port spacing. Adjust parameters."
        )

    # -------------------------------------------------
    # 7. Parameter label (TEXT layer, ignored by the EM model and the DRC)
    # -------------------------------------------------
    if textlabel == "":
        textlabel = (
            f"  bottom winding diameter: {bottom_winding_diameter:.2f}\n"
            f"  top winding diameter: {top_winding_diameter:.2f}\n"
            f"  center displacement: {2 * half_offset:.3f}\n"
            f"  bottom linewidth: {bottom_linewidth:.2f}\n"
            f"  top linewidth: {top_linewidth:.2f}\n"
            f"  bottom center tap width: {bottom_centertap_width if draw_bottom_tap else 0:.2f}\n"
            f"  top center tap width: {top_centertap_width if draw_top_tap else 0:.2f}\n"
            f"  bottom feed gap: {fs_bot:.2f}\n"
            f"  top feed gap: {fs_top:.2f}\n"
            f"  ground spacing (upper/lower/side): {gnd_upper_spacing:.2f} / "
            f"{gnd_lower_spacing:.2f} / {gnd_side_spacing:.2f}\n"
            f"  ground ring width: {gnd_ring_width:.2f}\n"
        )
    c.add_label(textlabel, position=(0.0, 0.0), layer=SG13G2.TEXT)

    # The windings are drawn on the grid with exact 45 degree sides; anything else the
    # DRCChecker stage snaps to the manufacturing grid (orca.geometry.drc).
    return c
