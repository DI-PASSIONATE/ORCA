#!/usr/bin/env python3
# build_tf_octa_c_ports.py
#
# Creates a SKILL-style (octagon-segment) transformer in GDS using gdsfactory,
# with Palace-compatible port marker layers 201..204 and automatic metal-layer
# numbers pulled from the setupEM substrate XML (SG13G2_nosub.xml).
#
# Usage:
#   python3 build_tf_octa_c_ports.py
#
# Output:
#   tf_octa_c_ports.gds  (in current working directory)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gdsfactory as gf

# Shapely is used to "stroke" a centerline into a conductor polygon robustly
from shapely.geometry import LineString, Polygon


def read_setupem_layer_map(substrate_xml: str) -> Dict[str, int]:
    """
    Returns a map {layername: gds_layer_number} extracted from setupEM substrate XML.
    """
    from gds2palace import stackup_reader  # must exist in your Palace/setupEM env

    _, _, metals_list = stackup_reader.read_substrate(substrate_xml)

    layer_map: Dict[str, int] = {}

    if hasattr(metals_list, "metals"):
        for m in metals_list.metals:
            name = getattr(m, "name", None)
            ln = getattr(m, "layernumber", None)
            if name is not None and ln is not None:
                layer_map[str(name)] = int(ln)

    if not layer_map:
        for attr in ("_metals", "layers", "stack", "data"):
            obj = getattr(metals_list, attr, None)
            if isinstance(obj, (list, tuple)):
                for m in obj:
                    name = getattr(m, "name", None)
                    ln = getattr(m, "layernumber", None)
                    if name is not None and ln is not None:
                        layer_map[str(name)] = int(ln)

    return layer_map


def octa_spiral_centerline(
    d_ff_start: float,
    pitch: float,
    n_turns: float,
    start_angle_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Generates a centerline polyline for an octagonal spiral.

    - d_ff_start: flat-to-flat diameter (centerline) of outer octagon at start
    - 8 segments per turn
    - side_len = 2*a*tan(pi/8), where a is apothem (flat-to-center)
    - apothem reduces by ~pitch per turn (distributed per segment)

    Units: microns.
    """
    a = d_ff_start / 2.0
    if a <= 0:
        raise ValueError("d_ff_start must be > 0")

    n_segments = max(1, int(round(n_turns * 8)))
    da = pitch / 8.0

    theta0 = math.radians(start_angle_deg)
    dtheta = math.radians(45.0)

    x, y = a, 0.0
    pts: List[Tuple[float, float]] = [(x, y)]

    for k in range(n_segments):
        theta = theta0 + k * dtheta
        side_len = 2.0 * a * math.tan(math.pi / 8.0)

        x += side_len * math.cos(theta)
        y += side_len * math.sin(theta)
        pts.append((x, y))

        a -= da
        if a <= 0:
            break

    return pts


def stroke_polyline_to_polygon(
    pts: List[Tuple[float, float]],
    width: float,
    join_style: int = 2,
    cap_style: int = 2,
) -> Polygon:
    """
    Uses shapely buffer to create a polygon conductor of given width around the centerline.
    """
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to stroke a polyline")
    if width <= 0:
        raise ValueError("width must be > 0")

    line = LineString(pts)
    poly = line.buffer(width / 2.0, join_style=join_style, cap_style=cap_style)
    if poly.is_empty:
        raise RuntimeError("Stroking produced empty polygon")
    return poly


def add_shapely_polygon_to_gf(component: gf.Component, poly: Polygon, layer: Tuple[int, int]) -> None:
    """
    Adds a shapely Polygon or MultiPolygon to a gdsfactory Component as polygons.
    """
    if poly.geom_type == "Polygon":
        component.add_polygon(list(poly.exterior.coords), layer=layer)
    else:
        for p in poly.geoms:
            component.add_polygon(list(p.exterior.coords), layer=layer)


@dataclass
class TfOctaParams:
    di: float = 50.0
    do: float = 50.0
    dis: float = 10.0
    wi: float = 5.0
    wo: float = 5.0
    si: float = 2.0
    so: float = 2.0
    turns_i: float = 1.5
    turns_o: float = 1.5
    port_w: float = 8.0
    port_l: float = 8.0


def centered_rect_points(cx: float, cy: float, w: float, l: float) -> List[Tuple[float, float]]:
    return [
        (cx - l / 2.0, cy - w / 2.0),
        (cx + l / 2.0, cy - w / 2.0),
        (cx + l / 2.0, cy + w / 2.0),
        (cx - l / 2.0, cy + w / 2.0),
    ]


def build_tf_octa_c_ports(
    params: TfOctaParams,
    L_TopMetal1: int,
    L_TopMetal2: int,
    L_Metal5: int,
    L_port1: int = 201,
    L_port2: int = 204,
    L_port3: int = 202,
    L_port4: int = 203,
) -> gf.Component:
    c = gf.Component("tf_octa_c_ports")

    tm1 = (int(L_TopMetal1), 0)
    tm2 = (int(L_TopMetal2), 0)
    m5  = (int(L_Metal5), 0)

    p1 = (int(L_port1), 0)
    p2 = (int(L_port2), 0)
    p3 = (int(L_port3), 0)
    p4 = (int(L_port4), 0)

    pitch_i = params.wi + params.si
    pitch_o = params.wo + params.so

    pts_i = octa_spiral_centerline(params.di, pitch_i, params.turns_i, start_angle_deg=0.0)
    pts_o = octa_spiral_centerline(params.do, pitch_o, params.turns_o, start_angle_deg=0.0)

    # simple deterministic separation
    dx_sep = params.dis / 2.0
    pts_o = [(x + dx_sep, y) for (x, y) in pts_o]
    pts_i = [(x - dx_sep, y) for (x, y) in pts_i]

    poly_i = stroke_polyline_to_polygon(pts_i, width=params.wi)
    poly_o = stroke_polyline_to_polygon(pts_o, width=params.wo)

    add_shapely_polygon_to_gf(c, poly_o, tm1)
    add_shapely_polygon_to_gf(c, poly_i, tm2)

    o_start, o_end = pts_o[0], pts_o[-1]
    i_start, i_end = pts_i[0], pts_i[-1]

    # port markers
    c.add_polygon(centered_rect_points(o_start[0], o_start[1], params.port_w, params.port_l), layer=p1)
    c.add_polygon(centered_rect_points(o_end[0],   o_end[1],   params.port_w, params.port_l), layer=p2)
    c.add_polygon(centered_rect_points(i_start[0], i_start[1], params.port_w, params.port_l), layer=p3)
    c.add_polygon(centered_rect_points(i_end[0],   i_end[1],   params.port_w, params.port_l), layer=p4)

    # small metal pads at port locations
    c.add_polygon(centered_rect_points(o_start[0], o_start[1], params.port_w, params.port_l), layer=tm1)
    c.add_polygon(centered_rect_points(o_end[0],   o_end[1],   params.port_w, params.port_l), layer=tm1)
    c.add_polygon(centered_rect_points(i_start[0], i_start[1], params.port_w, params.port_l), layer=tm2)
    c.add_polygon(centered_rect_points(i_end[0],   i_end[1],   params.port_w, params.port_l), layer=tm2)

    # optional Metal5 pads
    c.add_polygon(centered_rect_points(o_start[0], o_start[1], params.port_w, params.port_l), layer=m5)
    c.add_polygon(centered_rect_points(o_end[0],   o_end[1],   params.port_w, params.port_l), layer=m5)
    c.add_polygon(centered_rect_points(i_start[0], i_start[1], params.port_w, params.port_l), layer=m5)
    c.add_polygon(centered_rect_points(i_end[0],   i_end[1],   params.port_w, params.port_l), layer=m5)

    return c


if __name__ == "__main__":
    L_Metal5    = 58
    L_TopMetal1 = 110
    L_TopMetal2 = 111

    params = TfOctaParams(
        di=50.0, do=50.0, dis=10.0,
        wi=5.0, wo=5.0,
        si=1.0, so=1.0,
        turns_i=0.8, turns_o=0.8,
        port_w=8.0, port_l=8.0,
    )

    c = build_tf_octa_c_ports(params, L_TopMetal1, L_TopMetal2, L_Metal5)
    c.show()
    c.write_gds("tf_octa_c_ports.gds")
    print("Wrote tf_octa_c_ports.gds")
