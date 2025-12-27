import gdsfactory as gf
import numpy as np


def octagon_points(radius):
    """Returns octagon vertices"""
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]


@gf.cell
def asymmetric_transformer(
    top_radius=60.0,
    bottom_radius=45.0,
    top_width=4.0,
    bottom_width=2.5,
    top_layer=(1, 0),
    bottom_layer=(2, 0),
):
    """
    Coplanar asymmetric single-turn RF transformer

    Ports:
    - T_L, T_R : Top metal
    - B_L, B_R : Bottom metal
    """

    c = gf.Component()

    # --- Top winding ---
    top_path = gf.Path(octagon_points(top_radius))
    top_path.closed = True
    top_xs = gf.cross_section.strip(
        width=top_width,
        layer=top_layer,
    )
    top_ref = c.add_ref(top_path.extrude(top_xs))

    # --- Bottom winding ---
    bot_path = gf.Path(octagon_points(bottom_radius))
    bot_path.closed = True
    bot_xs = gf.cross_section.strip(
        width=bottom_width,
        layer=bottom_layer,
    )
    bot_ref = c.add_ref(bot_path.extrude(bot_xs))

    # --- Ports (left/right, asymmetric radii) ---
    c.add_port(
        name="T_L",
        center=(-top_radius, 0),
        width=top_width,
        orientation=180,
        layer=top_layer,
    )
    c.add_port(
        name="T_R",
        center=(top_radius, 0),
        width=top_width,
        orientation=0,
        layer=top_layer,
    )

    c.add_port(
        name="B_L",
        center=(-bottom_radius, 0),
        width=bottom_width,
        orientation=180,
        layer=bottom_layer,
    )
    c.add_port(
        name="B_R",
        center=(bottom_radius, 0),
        width=bottom_width,
        orientation=0,
        layer=bottom_layer,
    )

    return c


if __name__ == "__main__":
    c = asymmetric_transformer(
        top_radius=65,
        bottom_radius=42,
        top_width=5,
        bottom_width=2.5,
    )
    c.plot()
    c.show()
