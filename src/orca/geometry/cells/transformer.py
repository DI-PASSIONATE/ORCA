import gdsfactory as gf
import numpy as np
from ihp import PDK


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
        di: Diameter of input (lower/bot) winding (trace center to center).
        do: Diameter of output (upper/top) winding (trace center to center).
        dis: Displacement (offset between winding centers).
        wi: Trace width of lower winding.
        wic: Trace width of lower center tap.
        fi: Feed type lower (0=no, 1=fwd, 2=rev, 3=both).
        wo: Trace width of upper winding.
        woc: Trace width of upper center tap.
        fo: Feed type upper (0=no, 1=fwd, 2=rev, 3=both).
        fs: Feedline spacing (gap between inner sides of feed lines).
        ro: Ring spacing on upper winding side.
        ri: Ring spacing on lower winding side.
        rs: Ring spacing at side.
        rw: Ring width.
    """
    LAYER_BOT = PDK.layers.TopMetal1drawing
    LAYER_TOP = PDK.layers.TopMetal2drawing
    LAYER_RING = PDK.layers.Metal5drawing

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
    fo_int = int(round(upper_feed_type))
    fi_int = int(round(lower_feed_type))

    # --- Safety Check: Octagon Opening Width ---
    # Top Winding Gap is on the RIGHT.
    # Bot Center Tap (if fi_int & 1) goes RIGHT. It crosses Top Gap.
    fs_top = max(feedline_spacing, bottom_centertap_width)

    # Bot Winding Gap is on the LEFT.
    # Top Center Tap (if fo_int & 2) goes LEFT. It crosses Bot Gap.
    fs_bot = max(feedline_spacing, top_centertap_width)

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
    elif top_linewidth > top_winding_diameter / 3.0:
        raise ValueError("upper_linewidth is too large for output_winding_diameter.")
    # Check if center tap width is too large for winding diameter of the other winding
    if bottom_centertap_width > top_winding_diameter / 3.0:
        raise ValueError(
            "bottom_center_tap_width is too large for output_winding_diameter."
        )
    elif top_centertap_width > bottom_winding_diameter / 3.0:
        raise ValueError(
            "upper_center_tap_width is too large for input_winding_diameter."
        )
    elif abs(bottom_winding_diameter - top_winding_diameter) > 40.0:
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
    ):
        """
        Creates octagon winding AND the feed extension lines (rectangles) to the port.
        gap_size: spacing between inner edges of feed lines.
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

        # Create center tap - find point
        p_center_local = (round(x_end + width / 2.0, 2), round(center_y, 2))
        start_ct = transform(p_center_local)
        end_ct = (round(centertap_target_x, 2), round(center_y, 2))

        path_ct = gf.Path([start_ct, end_ct])
        c << path_ct.extrude(width=width, layer=layer)

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
        centertap_target_x=port_xl + gnd_ring_width,
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
        centertap_target_x=port_xr - gnd_ring_width,
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

    # Visual/meshing markers for ports: small rectangles on port layers so GDS contains geometry for source_layernum 201-204
    port_len = 1.0  # Length of port marker rectangles

    def add_port_marker(center, width, layer, thin_ports=False):
        rect = gf.components.rectangle(
            size=(port_len, width / 2.0 if thin_ports else width), layer=layer
        )
        ref = c << rect
        # ref.move((center[0], center[1] - width / 4.0))
        ref.move(
            (
                round(center[0], 2),
                round(center[1] - (width / 4.0 if thin_ports else width / 2.0), 2),
            )
        )

    ### TOP LAYER (ports on the RIGHT) -> Port 1 and 2 -> Layer 201, 202
    # OP (Top, Right, Upper)
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
    )
    # ON (Top, Right, Lower)
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
    )
    # Center Tap (Top, Center)
    c.add_port(
        name="oci",
        center=(round(port_xl + gnd_ring_width, 2), 0.0),
        width=top_centertap_width,
        orientation=180,
        layer=(205, 0),
    )
    add_port_marker(
        (round(port_xl + gnd_ring_width, 2), 0.0), top_centertap_width, (205, 0)
    )

    ### BOT LAYER (ports on the LEFT) -> Port 3 and 4 -> Layer 203, 204
    # IP (Bot, Left, Upper)
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
    )
    # IN (Bot, Left, Lower)
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
    )
    # Center Tap (Bot, Center)
    c.add_port(
        name="ico",
        center=(round(port_xr - gnd_ring_width, 2), 0.0),
        width=bottom_centertap_width,
        orientation=0,
        layer=(206, 0),
    )
    add_port_marker(
        (round(port_xr - gnd_ring_width, 2), 0.0), bottom_centertap_width, (206, 0)
    )

    # -------------------------------------------------
    # 5. Center Taps (ici, ico, oci, oco)
    # -------------------------------------------------
    # Top Winding (Layer Top)
    # "Left" edge of Top winding (Back of C-shape)
    # top_back_x = center_displacement/2.0 - output_winding_diameter/2.0

    # OCI (Top, goes Left?)
    # if fo_int & 1:

    # else:
    #     # Default placeholder port
    #     c.add_port(name="oci", center=(top_back_x - upper_linewidth/2.0 , 0), width=0.25, orientation=180, layer=(205, 0))
    #     add_port_marker((top_back_x - upper_linewidth/2.0 , 0), 0.25, (205, 0))

    # # OCO (Top, goes Right)
    # if fo_int & 2:
    #     c << gf.Path([(top_back_x, 0), (port_xr, 0)]).extrude(width=woc_int, layer=LAYER_TOP)
    #     c.add_port(name="oco", center=(port_xr, 0), width=woc_int, orientation=0, layer=(207, 0))
    #     add_port_marker((port_xr, 0), woc_int, (207, 0))
    # else:
    # c.add_port(name="oco", center=(top_back_x + upper_linewidth/2.0, 0), width=0.25, orientation=0, layer=(207, 0))
    # add_port_marker((top_back_x + upper_linewidth/2.0, 0), 0.25, (207, 0))

    # Bot Winding (Layer Bot)
    # bot_back_x = -center_displacement/2.0 + input_winding_diameter/2

    # ICO (Bot, goes Right)
    # if fi_int & 1:

    # else:
    #     c.add_port(name="ico", center=(bot_back_x + bottom_linewidth/2.0, 0), width=0.25, orientation=0, layer=(206, 0))
    #     add_port_marker((bot_back_x + bottom_linewidth/2.0, 0), 0.25, (206, 0))

    # # ICI (Bot, goes Left)
    # if fi_int & 2:
    #     c << gf.Path([(port_xl, 0), (bot_back_x, 0)]).extrude(width=wic_int, layer=LAYER_BOT)
    #     c.add_port(name="ici", center=(port_xl, 0), width=wic_int, orientation=180, layer=(208, 0))
    #     add_port_marker((port_xl, 0), wic_int, (208, 0))
    # else:
    #     c.add_port(name="ici", center=(bot_back_x - bottom_linewidth/2.0, 0), width=0.25, orientation=180, layer=(208, 0))
    #     add_port_marker((bot_back_x - bottom_linewidth/2.0, 0), 0.25, (208, 0))

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

    return c
