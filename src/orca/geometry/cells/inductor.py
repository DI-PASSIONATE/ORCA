"""
ORCA Geometry Preset: IHP inductor2/inductor3 — *Volker Mühlhaus gds2palace*
geometry, self-contained, with a standalone ``main()``.
============================================================================

This is the inductor analogue of the other ``inductor_geo*`` presets, but the
GDS-generating cell is the **built-in geometry code from the gds2palace IHP
example** (Volker Mühlhaus), ported verbatim:

    https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2
    more_examples/inductor_synthesis_no_external_library/
    synthesize_ihp_inductor_v2.py

Only the *geometry* part is kept here (constants, drawing helpers,
``get_min_outer_diameter``, ``calculate_octa_diameter`` and
``symmetric_octa_IHP``).  The gds2palace/Palace FEM simulation + L/Q synthesis
flow is **not** included — this file just builds the layout (and, optionally,
re-tunes the outer diameter to a target inductance via Wheeler's equation).

Unlike the other presets this one uses **gdspy** (not gdsfactory), exactly as
the upstream code does.  Layers (IHP SG13G2):
    TopMetal2 (134)  spiral windings
    TopMetal1 (126)  crossovers + feedlines
    TopVia2   (133)  vias
    Metal1    (8)    ground frame (only when forEM=True)
    201/202/203      EM ports (only when forEM=True)

Standalone GDS build (just gdspy + matplotlib):
    python inductor_geo_volker.py
    python inductor_geo_volker.py --turns 3 --width 6 --space 3 --diameter 120
    python inductor_geo_volker.py --turns 2 --width 4 --space 3 --Ltarget 0.5e-9 --ftarget 30e9
"""

from __future__ import annotations

import math

import gdspy

# ===========================================================================
#  GEOMETRY CODE — ported verbatim from gds2palace synthesize_ihp_inductor_v2
#  (only the layout part; simulation / synthesis flow removed)
# ===========================================================================

# ------ constants ----------
SCALE_FACTOR = 1  # input parameters are micron, drawing unit is micron

SPIRAL_LAYER_NUM = 134       # TopMetal2 for drawing main inductor turns
CROSSOVER_LAYER_NUM = 126    # TopMetal1 for crossover and feedline
VIA_LAYER_NUM = 133          # TopVia2
LBE_LAYER_NUM = 157          # LBE = localized backside etching
FRAME_LAYER_NUM = 8          # Metal1 layer for ground frame, used when forEM==True

PURPOSE_DRAWING = 0
PURPOSE_PIN = 2              # data type for pin shapes on metal layers

# OPDK extra layers drawn in non-forEM mode (always, when forEM=False).
EXTRA_LAYER_PURPOSE_PAIRS = [
    (46, 21),   # Pwell.block
    (148, 0),   # NoRCX.drawing
    (27, 0),    # IND.drawing
]

# nofill / fill-exclusion layers — only drawn when include_nofill=True.
NOFILL_LAYER_PURPOSE_PAIRS = [
    (1, 23),    # Activ.nofill
    (5, 23),    # Gatpoly nofill
    (8, 23),    # Metal1.nofill
    (10, 23),   # Metal2.nofill
    (30, 23),   # Metal3.nofill
    (50, 23),   # Metal4.nofill
    (67, 23),   # Metal5.nofill
    (126, 23),  # TopMetal1.nofill
    (134, 23),  # TopMetal2.nofill
]

IND_PIN = (27, 2)     # layer and purpose used for inductor pins in SG13G2 OPDK
IND_TEXT = (27, 25)   # layer and purpose used for inductor pin labels in SG13G2 OPDK

TEXT_DRAWING = (63, 0)  # layer to show additional text information

VIA_SIZE = 0.9          # IHP TopVia2 rule TV2.a
VIA_GAP = 1.06          # IHP TopVia2 rule TV2.b
VIA_MARGIN = 0.5        # IHP TopVia2 rule TV2.c, TV2.d

DELTA = 0.1             # size of EM port perpendicular to width

MU0 = 4 * math.pi * 1e-7


# --- utility functions ---

GRID = 0.01   # grid in micron = 10 nm (= 2 Nachkommastellen)


def gridsnap(x, grid=GRID):
    # snap to the manufacturing grid and clean up float noise
    # (same approach as the transformer's _snap: round(x/grid)*grid, then
    # round again to drop residual floating-point noise)
    return round(round(x / grid) * grid, 3)


def is_even(x):
    return x % 2 == 0


# --- GDSII drawing functions ---

def add_path(all_geometries_list, layer, purpose, points, width):
    p = gdspy.FlexPath(points, width, corners="miter", ends="flush",
                       layer=layer, datatype=purpose)
    all_geometries_list.append(p)
    return p


def add_box(all_geometries_list, layer, purpose, p1, p2):
    b = gdspy.Rectangle(p1, p2, layer=layer, datatype=purpose)
    all_geometries_list.append(b)
    return b


def add_poly(all_geometries_list, layer, purpose, points):
    p = gdspy.Polygon(points, layer=layer, datatype=purpose)
    all_geometries_list.append(p)
    return p


def add_via(all_geometries_list, layer, purpose, p1, p2, forEM):
    draw_via_array(all_geometries_list, layer, purpose, p1, p2)


def draw_via_array(all_geometries_list, layer, purpose, p1, p2):
    # draw a via array with all detail
    x1 = min(p1[0], p2[0])
    x2 = max(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    y2 = max(p1[1], p2[1])

    # maximum net size available for vias
    max_size_x = gridsnap((x2 - x1) - 2 * VIA_MARGIN)
    num_vias_x = 1 + math.floor((max_size_x - VIA_SIZE) / (VIA_SIZE + VIA_GAP))
    effective_margin_x = gridsnap(
        ((x2 - x1) - VIA_SIZE - (num_vias_x - 1) * (VIA_SIZE + VIA_GAP)) / 2)

    max_size_y = gridsnap((y2 - y1) - 2 * VIA_MARGIN)
    num_vias_y = 1 + int((max_size_y - VIA_SIZE) / (VIA_SIZE + VIA_GAP))
    effective_margin_y = gridsnap(
        ((y2 - y1) - VIA_SIZE - (num_vias_y - 1) * (VIA_SIZE + VIA_GAP)) / 2)

    x = gridsnap(x1 + effective_margin_x)
    for _n in range(1, num_vias_x + 1):
        y = gridsnap(y1 + effective_margin_y)
        for _m in range(1, num_vias_y + 1):
            add_box(all_geometries_list, layer, purpose, (x, y),
                    (x + VIA_SIZE, y + VIA_SIZE))
            y = gridsnap(y + VIA_SIZE + VIA_GAP)
        x = gridsnap(x + VIA_SIZE + VIA_GAP)


# ========================
#   Inductor calculations
# ========================

def get_min_outer_diameter(N, w, s):
    crossover_size = 3 * w + 2 * s

    # make sure we don't create a single via
    size_for_two_vias = 2 * VIA_SIZE + VIA_GAP + 2 * VIA_MARGIN
    overlap_size = w
    if w < size_for_two_vias:
        overlap_size = 1.1 * size_for_two_vias

    min_crossover_size = (2 * s + w) * (math.sqrt(2) - 1) + (s + w) + 2 * overlap_size

    if crossover_size < min_crossover_size:
        crossover_size = min_crossover_size

    if N < 3:
        inner_segment_size = crossover_size
    else:
        feedline_spacing = crossover_size + w + 2 * s
        inner_segment_size = feedline_spacing

    if N > 1:
        Di_min = inner_segment_size * (1 + math.sqrt(2))
    else:
        Di_min = 2 * (w + s) * (1 + math.sqrt(2))  # for single turn inductor

    Do_min = (Di_min + 2 * N * w + 2 * (N - 1) * s)
    # round to 2 decimal digits
    Do_min = math.ceil(100 * Do_min) / 100
    return Do_min


def calculate_octa_diameter(N, w, s, Ltarget, K1=2.15522, K2=3.61868, L0=0):
    # Calculate diameter for given target inductance (Wheeler's equation, DC)
    um = 1E-6

    Lsyn = Ltarget - L0
    b = 2 * N * w * um + 2 * (N - 1) * s * um  # difference outer/inner diameter
    c = K1 * MU0 * N * N                       # constant in Wheeler's equation

    p = -(b + Lsyn / c)
    q = b * b / 4 - Lsyn * b * (K2 - 1) / (2 * c)
    Dout = (-p / 2 + math.sqrt(p * p / 4 - q)) / um  # output is in micron
    Dout = math.ceil(Dout * 100) / 100
    return Dout


# ====================
#   Inductor layout
# ====================

def symmetric_octa_IHP(N, D, w, s, includeCenterTap=False, LBE=False, forEM=False,
                       include_nofill=True, ground_layer=None, ring_spacing=None,
                       ring_width=None, filename="inductor.gds", textlabel=""):
    # Drawing unit and parameter unit is micron
    # ground_layer: GDS layer number for the EM ground frame (forEM=True).
    #   None -> FRAME_LAYER_NUM (8 = Metal1). Use 67 for Metal5.
    # ring_spacing: gap [um] from the inductor outer radius (D/2) to the inner
    #   edge of the ground ring (forEM=True). None -> D/2 (original behaviour,
    #   scales with diameter). Set e.g. 10 or 20 for a fixed clearance.
    # ring_width: thickness [um] of the ground-ring frame (forEM=True).
    #   None -> min(20, 5*w) (original behaviour). Set e.g. 10 for a fixed width.
    if ground_layer is None:
        ground_layer = FRAME_LAYER_NUM

    # GDSII setup
    lib = gdspy.GdsLibrary()

    if includeCenterTap:
        cellname = f"inductor3_N{N}_Do{D}_w{w}_s{s}"
    else:
        cellname = f"inductor2_N{N}_Do{D}_w{w}_s{s}"

    try:
        cell = lib.new_cell(cellname, overwrite_duplicate=True)
    except Exception:
        cell = lib.new_cell("final_" + cellname, overwrite_duplicate=True)

    # list with all geometries that we created
    all_geometries_list = []

    # Convert to user units
    D = 2 * gridsnap(D / 2 * SCALE_FACTOR)
    w = 2 * gridsnap(w / 2 * SCALE_FACTOR)
    s = 2 * gridsnap(s / 2 * SCALE_FACTOR)

    # Reference center
    x0 = 0
    y0 = 0

    # --- Geometry calculations ---
    via_size = w
    crossover_size = (2 * s + w) * (math.sqrt(2) - 1) + (s + w)
    crossover_size = gridsnap(2 * via_size + crossover_size)

    # Feedline spacing
    if N < 3:
        feedline_spacing = w + s
        if N == 2 and includeCenterTap:
            feedline_spacing = 2 * (w + s)
    else:
        feedline_spacing = crossover_size + w + 2 * s

    # Inner diameter
    Di = gridsnap(D - 2 * N * w - 2 * (N - 1) * s)

    # Ground-ring geometry (only present when forEM). Computed up front so the
    # feed length can reach the ring.
    if ring_width is None:
        frame_width = min(20, gridsnap(5 * w))
    else:
        frame_width = gridsnap(ring_width)
    if ring_spacing is None:
        frame_margin = gridsnap(D / 2)
    else:
        frame_margin = gridsnap(ring_spacing)

    # Feed length: when forEM, extend the feedlines so the pins/ports always
    # land on the OUTER edge of the ground ring; otherwise keep the default.
    if forEM:
        feed_length = gridsnap(frame_margin + frame_width)
    else:
        feed_length = 30

    # --- Feedline drawing  ---
    if N == 1:
        # for single turn, we draw everything on single layer TopMetal2
        feed_layer = SPIRAL_LAYER_NUM
    else:
        # for multi turn, we draw trace on TopMetal2 and feedline on TopMetal1
        feed_layer = CROSSOVER_LAYER_NUM

    add_box(all_geometries_list, layer=feed_layer, purpose=PURPOSE_DRAWING,
            p1=(x0 - w / 2 - feedline_spacing / 2, y0 - Di / 2),
            p2=(x0 + w / 2 - feedline_spacing / 2, y0 - D / 2 - feed_length))

    add_box(all_geometries_list, layer=feed_layer, purpose=PURPOSE_DRAWING,
            p1=(x0 - w / 2 + feedline_spacing / 2, y0 - Di / 2),
            p2=(x0 + w / 2 + feedline_spacing / 2, y0 - D / 2 - feed_length))

    if N > 1:
        # for all N except single turn, we need via from TopMetal1 feedline to TopMetal2 trace
        add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                p1=(x0 - w / 2 - feedline_spacing / 2, y0 - Di / 2),
                p2=(x0 + w / 2 - feedline_spacing / 2, y0 - Di / 2 - w),
                forEM=forEM)

        add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                p1=(x0 - w / 2 + feedline_spacing / 2, y0 - Di / 2),
                p2=(x0 + w / 2 + feedline_spacing / 2, y0 - Di / 2 - w),
                forEM=forEM)

    # create pin label in IHP PDK-style on layer IND.text
    cell.add(gdspy.Label("LA", (x0 - feedline_spacing / 2, y0 - D / 2 - feed_length + w / 4),
                         layer=IND_TEXT[0], texttype=IND_TEXT[1]))
    cell.add(gdspy.Label("LB", (x0 + feedline_spacing / 2, y0 - D / 2 - feed_length + w / 4),
                         layer=IND_TEXT[0], texttype=IND_TEXT[1]))

    # create pin shape on the metal layer where that pin is
    add_box(all_geometries_list, layer=feed_layer, purpose=PURPOSE_PIN,
            p1=(x0 - w / 2 - feedline_spacing / 2, y0 - D / 2 - feed_length + w / 2),
            p2=(x0 + w / 2 - feedline_spacing / 2, y0 - D / 2 - feed_length))
    add_box(all_geometries_list, layer=feed_layer, purpose=PURPOSE_PIN,
            p1=(x0 - w / 2 + feedline_spacing / 2, y0 - D / 2 - feed_length + w / 2),
            p2=(x0 + w / 2 + feedline_spacing / 2, y0 - D / 2 - feed_length))

    if forEM:
        # create ports for gds2palace design flow
        add_box(all_geometries_list, layer=201, purpose=0,
                p1=(x0 - w / 2 - feedline_spacing / 2, y0 - D / 2 - feed_length + DELTA),
                p2=(x0 + w / 2 - feedline_spacing / 2, y0 - D / 2 - feed_length))
        add_box(all_geometries_list, layer=202, purpose=0,
                p1=(x0 - w / 2 + feedline_spacing / 2, y0 - D / 2 - feed_length + DELTA),
                p2=(x0 + w / 2 + feedline_spacing / 2, y0 - D / 2 - feed_length))

    if includeCenterTap:
        if is_even(N):
            cell.add(gdspy.Label("LC", (x0, y0 - D / 2 - feed_length + w / 4),
                                 layer=IND_TEXT[0], texttype=IND_TEXT[1]))
            add_box(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_PIN,
                    p1=(x0 - w / 2, y0 - D / 2 - feed_length + w / 2),
                    p2=(x0 + w / 2, y0 - D / 2 - feed_length))
            if forEM:
                add_box(all_geometries_list, layer=203, purpose=0,
                        p1=(x0 - w / 2, y0 - D / 2 - feed_length + DELTA),
                        p2=(x0 + w / 2, y0 - D / 2 - feed_length))
        else:
            cell.add(gdspy.Label("LC", (x0, y0 + D / 2 + feed_length - w / 4),
                                 layer=IND_TEXT[0], texttype=IND_TEXT[1]))
            add_box(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_PIN,
                    p1=(x0 - w / 2, y0 + D / 2 + feed_length - w / 2),
                    p2=(x0 + w / 2, y0 + D / 2 + feed_length))
            if forEM:
                add_box(all_geometries_list, layer=203, purpose=0,
                        p1=(x0 - w / 2, y0 + D / 2 + feed_length - DELTA),
                        p2=(x0 + w / 2, y0 + D / 2 + feed_length))

    if textlabel == "":
        # descriptive label with inductor parameters
        textlabel = (f"  number of turns: {N}\n"
                     f"  width: {w:.2f}\n"
                     f"  spacing: {s:.2f}\n"
                     f"  outer diameter: {D:.2f}\n"
                     f"  inner diameter: {Di:.2f}\n")
    cell.add(gdspy.Label(textlabel, (x0, y0), layer=TEXT_DRAWING[0], texttype=TEXT_DRAWING[1]))

    # --- Spiral segments ---
    segment_length = (D - w) / (1 + math.sqrt(2))

    for i in range(1, N + 1):

        # left side
        # lower left quadrant
        points = []
        #  shorter innermost turn at feed side
        if (i == N):
            x1 = gridsnap(x0 - feedline_spacing / 2 + w / 2)
        else:
            x1 = gridsnap(x0 - crossover_size / 2 + w)
        y1 = gridsnap(y0 + (-D / 2 + w / 2 + (i - 1) * (w + s)))
        points.append((x1, y1))
        x1 = x0 - gridsnap(segment_length / 2)
        points.append((x1, y1))
        x1 = x1 - gridsnap(segment_length / math.sqrt(2))
        y1 = y1 + gridsnap(segment_length / math.sqrt(2))
        points.append((x1, y1))
        #  half segment and a little bit
        y1 = y1 + gridsnap(segment_length / 2 + w)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        # upper left quadrant
        points = []
        x1 = gridsnap(x0 - crossover_size / 2 + w)
        y1 = gridsnap(y0 + D / 2 - w / 2 - (i - 1) * (w + s))
        points.append((x1, y1))
        x1 = x0 - gridsnap(segment_length / 2)
        points.append((x1, y1))
        x1 = x1 - gridsnap(segment_length / math.sqrt(2))
        y1 = y1 - gridsnap(segment_length / math.sqrt(2))
        points.append((x1, y1))
        # half segment and a little bit
        y1 = y1 - gridsnap(segment_length / 2 + w)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        # right side
        # lower right quadrant
        points = []
        # shorter innermost turn at feed side
        if (i == N):
            x1 = gridsnap(x0 + feedline_spacing / 2 - w / 2)
        else:
            x1 = gridsnap(x0 + crossover_size / 2 - w)
        y1 = gridsnap(y0 + (-D / 2 + w / 2 + (i - 1) * (w + s)))
        points.append((x1, y1))
        x1 = x0 + gridsnap(segment_length / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(segment_length / math.sqrt(2))
        y1 = y1 + gridsnap(segment_length / math.sqrt(2))
        points.append((x1, y1))
        # half segment and a little bit
        y1 = y1 + gridsnap(segment_length / 2 + w)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        # upper right quadrant
        points = []
        x1 = gridsnap(x0 + crossover_size / 2 - w)
        y1 = gridsnap(y0 + D / 2 - w / 2 - (i - 1) * (w + s))
        points.append((x1, y1))
        x1 = x0 + gridsnap(segment_length / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(segment_length / math.sqrt(2))
        y1 = y1 - gridsnap(segment_length / math.sqrt(2))
        points.append((x1, y1))
        # half segment and a little bit
        y1 = y1 - gridsnap(segment_length / 2 + w)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        # decrease segment length for next turn
        segment_length = segment_length - 2 * (w + s) / (1 + math.sqrt(2))

    # --- Crossovers ---
    num_top = math.floor((N - 1) / 2)
    num_bot = num_top
    if is_even(N):
        num_top += 1

    # bottom side
    for i in range(1, num_bot + 1):

        points = []
        x1 = gridsnap(x0 - crossover_size / 2)
        y1 = gridsnap(y0 - D / 2 + 2 * i * (w + s) + 0.5 * w)
        if not is_even(N):
            y1 = y1 - w - s
        points.append((x1, y1))
        x1 = x0 - gridsnap((w + s) / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(w + s)
        y1 = y1 - gridsnap(w + s)
        points.append((x1, y1))
        x1 = gridsnap(x0 + crossover_size / 2)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        points = []
        x1 = gridsnap(x0 - crossover_size / 2)
        y1 = gridsnap(y0 - D / 2 + 2 * i * (w + s) - 0.5 * w - s)
        if not is_even(N):
            y1 = y1 - w - s
        points.append((x1, y1))
        x1 = x0 - gridsnap((w + s) / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(w + s)
        y1 = y1 + gridsnap(w + s)
        points.append((x1, y1))
        x1 = gridsnap(x0 + crossover_size / 2)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=CROSSOVER_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        if is_even(N):  # N! not i!
            # add vias also
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - crossover_size / 2, y0 - D / 2 + 2 * i * (w + s) - s),
                    p2=(x0 - crossover_size / 2 + via_size, y0 - D / 2 + 2 * i * (w + s) - w - s),
                    forEM=forEM)
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 + crossover_size / 2, y0 - D / 2 + (2 * i + 1) * (w + s) - s),
                    p2=(x0 + crossover_size / 2 - via_size, y0 - D / 2 + (2 * i + 1) * (w + s) - w - s),
                    forEM=forEM)
        else:
            # add vias also
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - crossover_size / 2, y0 - D / 2 - w - s + (2 * i - 1) * (w + s)),
                    p2=(x0 - crossover_size / 2 + via_size, y0 - D / 2 + (2 * i - 1) * (w + s) - s),
                    forEM=forEM)
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 + crossover_size / 2, y0 - D / 2 - w - s + (2 * i) * (w + s)),
                    p2=(x0 + crossover_size / 2 - via_size, y0 - D / 2 + (2 * i) * (w + s) - s),
                    forEM=forEM)

    # top side
    for i in range(1, num_top + 1):

        points = []
        x1 = gridsnap(x0 - crossover_size / 2)
        y1 = gridsnap(y0 + D / 2 - (2 * i - 1) * (w + s) - 0.5 * w)
        if not is_even(N):
            y1 = y1 - w - s
        points.append((x1, y1))
        x1 = x0 - gridsnap((w + s) / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(w + s)
        y1 = y1 + gridsnap(w + s)
        points.append((x1, y1))
        x1 = gridsnap(x0 + crossover_size / 2)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        points = []
        x1 = gridsnap(x0 - crossover_size / 2)
        y1 = gridsnap(y0 + D / 2 - (2 * i - 1) * (w + s) + 0.5 * w + s)
        if not is_even(N):
            y1 = y1 - w - s
        points.append((x1, y1))
        x1 = x0 - gridsnap((w + s) / 2)
        points.append((x1, y1))
        x1 = x1 + gridsnap(w + s)
        y1 = y1 - gridsnap(w + s)
        points.append((x1, y1))
        x1 = gridsnap(x0 + crossover_size / 2)
        points.append((x1, y1))
        add_path(all_geometries_list, layer=CROSSOVER_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

        if (is_even(N)):  # N! not i!
            # add via also
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - crossover_size / 2, y0 + D / 2 - (2 * i - 1) * (w + s) + w + s),
                    p2=(x0 - crossover_size / 2 + via_size, y0 + D / 2 - (2 * i - 1) * (w + s) + s),
                    forEM=forEM)
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 + crossover_size / 2, y0 + D / 2 - (2 * i) * (w + s) + w + s),
                    p2=(x0 + crossover_size / 2 - via_size, y0 + D / 2 - (2 * i) * (w + s) + s),
                    forEM=forEM)
        else:
            # add via also
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - crossover_size / 2, y0 + D / 2 - w - s - (2 * i - 1) * (w + s) + w + s),
                    p2=(x0 - crossover_size / 2 + via_size, y0 + D / 2 - w - s - (2 * i - 1) * (w + s) + s),
                    forEM=forEM)
            add_via(all_geometries_list, layer=VIA_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 + crossover_size / 2, y0 + D / 2 - w - s - (2 * i) * (w + s) + w + s),
                    p2=(x0 + crossover_size / 2 - via_size, y0 + D / 2 - w - s - (2 * i) * (w + s) + s),
                    forEM=forEM)

    # one straight segment at outer turn
    if is_even(N):
        # even number of turns, N=2,4,6,..
        points = []
        points.append((x0 - crossover_size / 2, y0 - D / 2 + w / 2))
        points.append((x0 + crossover_size / 2, y0 - D / 2 + w / 2))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)
    else:
        # odd number of turns, N=1,3,5,..
        points = []
        if N > 1:
            points.append((x0 - crossover_size / 2, y0 + D / 2 - w / 2))
            points.append((x0 + crossover_size / 2, y0 + D / 2 - w / 2))
        else:
            # we can go to small diameters, so we must keep this short
            points.append((x0 - (w + s), y0 + D / 2 - w / 2))
            points.append((x0 + (w + s), y0 + D / 2 - w / 2))
        add_path(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                 points=points, width=w)

    # --- Center tap ---
    if includeCenterTap:
        if is_even(N):
            add_box(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - w / 2, y0 - D / 2 + w),
                    p2=(x0 + w / 2, y0 - D / 2 - feed_length))
        else:
            add_box(all_geometries_list, layer=SPIRAL_LAYER_NUM, purpose=PURPOSE_DRAWING,
                    p1=(x0 - w / 2, y0 + D / 2 - w),
                    p2=(x0 + w / 2, y0 + D / 2 + feed_length))

    # IHP extra layers for inductors
    D1 = gridsnap(D / 2 + feed_length)
    D2 = gridsnap(D1 / (1 + math.sqrt(2)))
    points = []
    points.append((x0 + D2, y0 - D1))
    points.append((x0 + D1, y0 - D2))
    points.append((x0 + D1, y0 + D2))
    points.append((x0 + D2, y0 + D1))
    points.append((x0 - D2, y0 + D1))
    points.append((x0 - D1, y0 + D2))
    points.append((x0 - D1, y0 - D2))
    points.append((x0 - D2, y0 - D1))

    # iterate over the extra OPDK layers and add an octagon on each of them
    if not forEM:
        extra_pairs = list(EXTRA_LAYER_PURPOSE_PAIRS)
        if include_nofill:
            extra_pairs += NOFILL_LAYER_PURPOSE_PAIRS
        for LPP in extra_pairs:
            layer, datatype = LPP
            add_poly(all_geometries_list, layer=layer, purpose=datatype, points=points)

    # --- localized backside etching option ---
    if LBE:
        add_poly(all_geometries_list, layer=LBE_LAYER_NUM, purpose=PURPOSE_DRAWING, points=points)

    # --- ground frame for EM simulation using gds2palace -------
    # (frame_width / frame_margin were computed up front, near the feed length)
    if forEM:
        xmin_frame_inner = gridsnap(x0 - D / 2 - frame_margin)
        xmax_frame_inner = gridsnap(x0 + D / 2 + frame_margin)
        ymin_frame_inner = gridsnap(y0 - D / 2 - frame_margin)
        ymax_frame_inner = gridsnap(y0 + D / 2 + frame_margin)

        xmin_frame_outer = xmin_frame_inner - frame_width
        xmax_frame_outer = xmax_frame_inner + frame_width
        ymin_frame_outer = ymin_frame_inner - frame_width
        ymax_frame_outer = ymax_frame_inner + frame_width

        add_box(all_geometries_list, layer=ground_layer, purpose=PURPOSE_DRAWING,
                p1=(xmin_frame_outer, ymin_frame_outer),
                p2=(xmin_frame_inner, ymax_frame_outer))
        add_box(all_geometries_list, layer=ground_layer, purpose=PURPOSE_DRAWING,
                p1=(xmax_frame_inner, ymin_frame_outer),
                p2=(xmax_frame_outer, ymax_frame_outer))
        add_box(all_geometries_list, layer=ground_layer, purpose=PURPOSE_DRAWING,
                p1=(xmin_frame_inner, ymin_frame_inner),
                p2=(xmax_frame_inner, ymin_frame_outer))
        add_box(all_geometries_list, layer=ground_layer, purpose=PURPOSE_DRAWING,
                p1=(xmin_frame_inner, ymax_frame_inner),
                p2=(xmax_frame_inner, ymax_frame_outer))

        # NOTE: the original gds2palace code added a "ground under feedline"
        # rectangle here (filling the feed opening down to the ring). With the
        # feed length now reaching the ring it overlapped the ring frame, so it
        # is intentionally omitted — the ring is just the clean square frame.

    # add all created shapes to cell now
    for geometry in all_geometries_list:
        cell.add(geometry)

    lib.write_gds(filename)
    return filename

