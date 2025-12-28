import gdsfactory as gf
import numpy as np

# Activate IHP PDK
from ihp import PDK
PDK.activate()

def octagon(
    radius: float,
    width: float,
    lead_length: float | None = None,
    stub_length: float | None = None,
    shift_amount: float = 0.0,
):
    """Return polygons plus lead metadata for an octagonal ring.

    Rectangles extend from the open end to facilitate port connections and from
    the closed end to provide a stub. Overlap between shapes is acceptable.
    """

    if width <= 0:
        raise ValueError("width must be positive")

    if width >= radius:
        raise ValueError("width must be smaller than radius")

    lead = lead_length if lead_length is not None else width * 2.5
    stub = stub_length if stub_length is not None else width * 2 + shift_amount

    if lead <= 0 or stub <= 0:
        raise ValueError("lead_length and stub_length must be positive")

    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8

    outer = [
        (radius * np.cos(a), radius * np.sin(a))
        for a in angles
    ]

    inner = [
        ((radius - width) * np.cos(a), (radius - width) * np.sin(a))
        for a in reversed(angles)
    ]

    polygons = [outer + inner]
    lead_rectangles = []

    lead_points = sorted(outer, key=lambda pt: pt[0], reverse=True)[:2]

    for px, py in lead_points:
        x0 = px - width
        x1 = x0 + lead
        if py >= 0:
            y0 = py - width
            y1 = py
        else:
            y0 = py
            y1 = py + width
        rect = [
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (x0, y1),
        ]
        polygons.append(rect)
        lead_rectangles.append(
            {
                "polygon": rect,
                "port_center": (x1, (y0 + y1) / 2),
                "port_width": abs(y1 - y0),
            }
        )

    stub_points = sorted(outer, key=lambda pt: pt[0])[:2]
    stub_center_y = sum(pt[1] for pt in stub_points) / 2
    stub_x1 = min(pt[0] for pt in stub_points) + width / 2
    stub_x0 = stub_x1 - stub
    stub_y0 = stub_center_y - width / 2
    stub_y1 = stub_center_y + width / 2
    polygons.append([
        (stub_x0, stub_y0),
        (stub_x1, stub_y0),
        (stub_x1, stub_y1),
        (stub_x0, stub_y1),
    ])

    return polygons, lead_rectangles


@gf.cell
def octagon_transformer(
    radius: float = 30.0,
    width: float = 7.0,
    shift: float = 10.0,
    lead_length: float | None = None,
    stub_length: float | None = None,
    layer_top=(PDK.layers.TopMetal2drawing, 0),
    layer_bottom=(PDK.layers.TopMetal1drawing, 0),
):
    """
    Octagonal transformer with two stacked, slightly shifted inductors.
    """

    c = gf.Component()

    base_polygons, lead_rectangles = octagon(
        radius=radius,
        width=width,
        lead_length=lead_length,
        stub_length=stub_length,
        shift_amount=shift,
    )

    # Bottom octagon (mirrored across the y-axis)
    for poly in base_polygons:
        mirrored = [(-x, y) for (x, y) in poly]
        c.add_polygon(mirrored, layer=layer_bottom)

    # Top octagon (shifted)
    for poly in base_polygons:
        shifted_poly = [(x + shift, y) for (x, y) in poly]
        c.add_polygon(shifted_poly, layer=layer_top)

    # Ports located at the end of the lead rectangles (non-stub)
    num_leads = len(lead_rectangles)
    port_layers = [(201 + i, 0) for i in range(num_leads * 2)]

    for idx, rect in enumerate(lead_rectangles):
        cx, cy = rect["port_center"]
        port_width = rect["port_width"]

        c.add_port(
            name=f"B{idx+1}",
            center=(-cx, cy),
            width=port_width,
            orientation=180,
            layer=port_layers[idx],
        )

    for idx, rect in enumerate(lead_rectangles):
        cx, cy = rect["port_center"]
        port_width = rect["port_width"]
        port_idx = num_leads + idx

        c.add_port(
            name=f"T{idx+1}",
            center=(cx + shift, cy),
            width=port_width,
            orientation=0,
            layer=port_layers[port_idx],
        )

    return c


if __name__ == "__main__":
    c = octagon_transformer()
    c.show()

    # Export to GDS
    c.write_gds("octagon_transformer.gds")
