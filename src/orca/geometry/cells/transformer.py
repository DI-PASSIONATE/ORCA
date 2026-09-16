import gdsfactory as gf
import klayout.db as kdb
import numpy as np

from orca.geometry.layers import SG13G2

GRID_NM = 10  # 10 nm manufacturing grid = 0.01 µm


def _ensure_active_pdk() -> None:
    """Gdsfactory refuses to extrude paths without an active PDK. We only use it as a
    polygon generator with explicit SG13G2 layer tuples, so its generic PDK is enough.
    """
    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

def _snap_inplace(c: gf.Component) -> None:
    """Snap all polygon vertices to GRID_NM across the component hierarchy.

    gf.Path.extrude() computes perpendicular offsets for angled segments
    (e.g. the 22.5° octagon sides) that land off the 10 nm grid.
    This iterates every cell in the component's call tree and snaps
    each polygon vertex in-place — without flattening, so the cutout
    geometry and all reference offsets stay untouched.
    """
    layout = c.layout()
    all_cell_idxs = set(c.called_cells())
    all_cell_idxs.add(c.cell_index())

    for idx in all_cell_idxs:
        cell = layout.cell(idx)
        for layer_idx in layout.layer_indexes():
            for shape in cell.shapes(layer_idx).each(kdb.Shapes.SPolygons):
                pts = [
                    kdb.Point(
                        round(p.x / GRID_NM) * GRID_NM,
                        round(p.y / GRID_NM) * GRID_NM,
                    )
                    for p in shape.polygon.each_point_hull()
                ]
                shape.polygon = kdb.Polygon(pts)

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
    gnd_upper_spacing: float = 10.0,
    gnd_lower_spacing: float = 10.0,
    gnd_side_spacing: float = 10.0,
    gnd_ring_width: float = 10.0,
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
    """
    _ensure_active_pdk()

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
    for label, feed_type in (("lower_feed_type", lower_feed_type), ("upper_feed_type", upper_feed_type)):
        if feed_type not in (0, 1):
            raise NotImplementedError(
                f"{label}={feed_type}: only 0 (no center tap) and 1 (center tap) are implemented."
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

    # X Limits for Ports
    # Note: Winding edges are approx at center +/- diameter/2
    top_right_x = (center_displacement / 2.0) + (top_winding_diameter / 2.0)
    bot_right_x = (-center_displacement / 2.0) + (bottom_winding_diameter / 2.0)
    port_xr = max(top_right_x, bot_right_x) + gnd_upper_spacing

    top_left_x = (center_displacement / 2.0) - (top_winding_diameter / 2.0)
    bot_left_x = (-center_displacement / 2.0) - (bottom_winding_diameter / 2.0)
    port_xl = min(top_left_x, bot_left_x) - gnd_lower_spacing

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
    if abs(bottom_winding_diameter - top_winding_diameter) > 40.0:
        raise ValueError(
            "input_winding_diameter and output_winding_diameter difference is too large. No sufficient coupling."
        )

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
        Creates octagon winding AND the feed extension lines (rectangles) to the port.
        gap_size: spacing between inner edges of feed lines.
        centertap_target_x: x of the center tap port, or None for no center tap.
        """
        r = diameter / 2.0
        # Feed Y positions (Trace Centers)
        # Inner edge is at +gap_size/2 and -gap_size/2
        y_upper = gap_size / 2.0
        y_lower = -gap_size / 2.0

        # Calculate start X (at the gap face)
        # Using 22.5 deg vertex logic (vertical flat side approximation)
        x_face = r * np.cos(np.radians(22.5))

        # Calculate end X (for center tap) on the other side
        # Using 22.5 deg vertex logic (vertical flat side approximation)
        x_end = -r * np.cos(np.radians(22.5))

        # --- 1. Main Octagon Path ---
        path_pts = []
        path_pts.append((round(x_face, 2), round(y_upper, 2)))  # Start at upper feed

        # Vertices (22.5 to 337.5)
        for ang in np.arange(22.5, 360, 45):
            rad = np.radians(ang)
            path_pts.append((round(r * np.cos(rad), 2), round(r * np.sin(rad), 2)))

        path_pts.append((round(x_face, 2), round(y_lower, 2)))  # End at lower feed

        p = gf.Path(path_pts)
        ref = c << p.extrude(width=width, layer=layer)
        ref.rotate(rotation_deg)
        ref.move((center_x, center_y))

        # --- 2. Feed Extensions (The "Small Rectangles") ---
        # These connect the winding gap to the boundary (port_x).
        # We need to calculate coordinates in the rotated frame or global frame.

        # Local coordinates of feed tips:
        p_up_local = (round(x_face - width / 2.0, 2), round(y_upper + width / 2.0, 2))
        p_lo_local = (round(x_face - width / 2.0, 2), round(y_lower - width / 2.0, 2))

        # Transform to Global
        # Rotation Matrix
        theta = np.radians(rotation_deg)
        c_rot, s_rot = np.cos(theta), np.sin(theta)

        def transform(pt):
            x, y = pt
            x_new = x * c_rot - y * s_rot + center_x
            y_new = x * s_rot + y * c_rot + center_y
            return (round(x_new, 2), round(y_new, 2))

        start_up = transform(p_up_local)
        start_lo = transform(p_lo_local)

        # End points are at feed_target_x with same Y
        end_up = (round(feed_target_x, 2), start_up[1])
        end_lo = (round(feed_target_x, 2), start_lo[1])

        # Create Feed Rectangles (Wires)
        # Upper Feed
        path_u = gf.Path([start_up, end_up])
        c << path_u.extrude(width=width, layer=layer)

        # Lower Feed
        path_l = gf.Path([start_lo, end_lo])
        c << path_l.extrude(width=width, layer=layer)

        # Center tap (optional): from the back of the winding straight out to its port
        if centertap_target_x is not None:
            p_center_local = (round(x_end + width / 2.0, 2), round(center_y, 2))
            start_ct = transform(p_center_local)
            end_ct = (round(centertap_target_x, 2), round(center_y, 2))

            path_ct = gf.Path([start_ct, end_ct])
            c << path_ct.extrude(width=centertap_width, layer=layer)

        return start_up, start_lo  # Return actual start points for reference if needed

    # -------------------------------------------------
    # 3. Create Geometry
    # -------------------------------------------------

    # Top Winding (Rot 0, Gap Right -> connects to port_xr)
    create_octa_winding(
        diameter=top_winding_diameter,
        width=top_linewidth,
        gap_size=fs_top,
        layer=LAYER_TOP,
        center_x=center_displacement / 2.0,
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
        center_x=-center_displacement / 2.0,
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
        center=(round(port_xr - gnd_ring_width, 2), round(y_top_p, 2)),
        width=top_linewidth,
        orientation=0,
        layer=(201, 0),
    )
    add_port_marker(
        (round(port_xr - gnd_ring_width, 2), round(y_top_p, 2)),
        top_linewidth,
        (201, 0),
        0,
    )
    # ON: top winding, right side, lower port
    c.add_port(
        name="on",
        center=(round(port_xr - gnd_ring_width, 2), round(y_top_n, 2)),
        width=top_linewidth,
        orientation=0,
        layer=(202, 0),
    )
    add_port_marker(
        (round(port_xr - gnd_ring_width, 2), round(y_top_n, 2)),
        top_linewidth,
        (202, 0),
        0,
    )
    # Center Tap (Top, Center) -> Layer 205
    if draw_top_tap:
        c.add_port(
            name="oci",
            center=(round(port_xl + gnd_ring_width, 2), 0.0),
            width=top_centertap_width,
            orientation=180,
            layer=(205, 0),
        )
        add_port_marker(
            (round(port_xl + gnd_ring_width, 2), 0.0), top_centertap_width, (205, 0), 180
        )

    ### BOT LAYER (ports on the LEFT) -> Port 3 and 4 -> Layer 203, 204
    # IP: bottom winding, left side, upper port
    c.add_port(
        name="ip",
        center=(round(port_xl + gnd_ring_width, 2), round(y_bot_p, 2)),
        width=bottom_linewidth,
        orientation=180,
        layer=(203, 0),
    )
    add_port_marker(
        (round(port_xl + gnd_ring_width, 2), round(y_bot_p, 2)),
        bottom_linewidth,
        (203, 0),
        180,
    )
    # IN: bottom winding, left side, lower port
    c.add_port(
        name="in",
        center=(round(port_xl + gnd_ring_width, 2), round(y_bot_n, 2)),
        width=bottom_linewidth,
        orientation=180,
        layer=(204, 0),
    )
    add_port_marker(
        (round(port_xl + gnd_ring_width, 2), round(y_bot_n, 2)),
        bottom_linewidth,
        (204, 0),
        180,
    )
    # Center Tap (Bot, Center) -> Layer 206
    if draw_bottom_tap:
        c.add_port(
            name="ico",
            center=(round(port_xr - gnd_ring_width, 2), 0.0),
            width=bottom_centertap_width,
            orientation=0,
            layer=(206, 0),
        )
        add_port_marker(
            (round(port_xr - gnd_ring_width, 2), 0.0), bottom_centertap_width, (206, 0), 0
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
        top_ref.move((round(port_xl, 2), round(tf_y - gnd_ring_width, 2)))

        # Bottom bar
        bot = gf.components.rectangle(size=(outer_w, gnd_ring_width), layer=LAYER_RING)
        bot_ref = c << bot
        bot_ref.move((round(port_xl, 2), round(-tf_y, 2)))

        # Left bar
        left_h = outer_h - 2 * gnd_ring_width
        if left_h > 0:
            left = gf.components.rectangle(
                size=(gnd_ring_width, left_h), layer=LAYER_RING
            )
            left_ref = c << left
            left_ref.move((round(port_xl, 2), round(-tf_y + gnd_ring_width, 2)))

        # Right bar
        if left_h > 0:
            right = gf.components.rectangle(
                size=(gnd_ring_width, left_h), layer=LAYER_RING
            )
            right_ref = c << right
            right_ref.move(
                (round(port_xr - gnd_ring_width, 2), round(-tf_y + gnd_ring_width, 2))
            )
    else:
        raise ValueError(
            "Ground ring dimensions are invalid due to port spacing. Adjust parameters."
        )

    # Snap all polygon vertices to the 10 nm manufacturing grid.
    # gf.Path.extrude() produces off-grid vertices for angled segments.
    # _snap_inplace iterates the component's cell hierarchy without flattening,
    # so the cutout geometry and reference offsets stay untouched.
    _snap_inplace(c)

    return c
