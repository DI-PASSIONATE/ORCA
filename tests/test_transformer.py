"""Tests for the stacked octagonal transformer cell and its preset."""

from __future__ import annotations

import itertools

import klayout.db as kdb
import pytest

from orca.geometry.cells.transformer import check_tf_octa_c_parameters, tf_octa_c
from orca.geometry.drc import check_gds_file
from orca.geometry.layers import SG13G2
from orca.geometry.presets import TransformerOcta

_names = itertools.count()

SMALLEST = {
    "bottom_winding_diameter": 20.0,
    "top_winding_diameter": 20.0,
    "relative_displacement": 0.0,
    "bottom_linewidth": 2.0,
    "top_linewidth": 2.0,
}


def _draw(tmp_path, params: dict) -> str:
    """Lay out the preset for *params* and return the GDS path."""
    name = f"tf_test_{next(_names)}"
    path = str(tmp_path / f"{name}.gds")
    TransformerOcta.create_gds_file(name, path, params)
    return path


def _metal(path: str, layer: tuple[int, int]) -> kdb.Region:
    layout = kdb.Layout()
    layout.read(path)
    return kdb.Region(layout.top_cell().begin_shapes_rec(layout.find_layer(*layer))).merged()


@pytest.mark.parametrize(
    "params",
    [
        SMALLEST,
        SMALLEST | {"bottom_winding_diameter": 24.0, "top_winding_diameter": 24.0,
                    "bottom_linewidth": 8.0, "top_linewidth": 8.0},
        SMALLEST | {"bottom_winding_diameter": 40.0, "top_winding_diameter": 30.0,
                    "relative_displacement": 0.2, "bottom_linewidth": 4.0, "top_linewidth": 3.0},
        SMALLEST | {"bottom_winding_diameter": 90.0, "top_winding_diameter": 90.0,
                    "bottom_linewidth": 8.0, "top_linewidth": 6.0},
    ],
    ids=["smallest", "small-wide-lines", "ratio-and-offset-limits", "largest"],
)
def test_preset_corners_are_feasible_and_drc_clean(tmp_path, params):
    assert TransformerOcta().is_feasible(params)
    result = check_gds_file(_draw(tmp_path, params))

    assert result.clean, result.violations
    # Drawn on the grid with exact 45 degree sides, nothing is left to snap
    assert result.snapped_vertices == 0


def test_sampled_layouts_are_drc_clean(tmp_path):
    geometry = TransformerOcta()
    iterator = geometry.input_parameter_iterator
    iterator.set_sample_count(40, seed=3, feasible=geometry.is_feasible)

    failures = {}
    for params in iterator:
        result = check_gds_file(_draw(tmp_path, params))
        if not result.clean:
            failures[str(params)] = result.violations

    assert not failures


def test_each_winding_with_its_feeds_and_tap_is_one_polygon(tmp_path):
    path = _draw(tmp_path, SMALLEST)

    for layer in (SG13G2.TopMetal1, SG13G2.TopMetal2):
        assert _metal(path, layer).count() == 1


def test_feed_gap_wider_than_the_flat_side_is_rejected():
    # A 20 µm octagon's flat side is 20 * sin(22.5 deg) = 7.65 µm along the centre line
    check_tf_octa_c_parameters(bottom_winding_diameter=20.0, top_winding_diameter=20.0,
                               bottom_linewidth=2.0, top_linewidth=2.0, feedline_spacing=7.6)
    with pytest.raises(ValueError, match="feed gap"):
        check_tf_octa_c_parameters(bottom_winding_diameter=20.0, top_winding_diameter=20.0,
                                   bottom_linewidth=2.0, top_linewidth=2.0,
                                   feedline_spacing=7.7)


def test_gap_beyond_the_inner_flat_side_draws_without_folding(tmp_path):
    # The gap reaches past the inner octagon's flat side (2 * (10.09 - 3) * tan 22.5 deg =
    # 5.9 µm), where the old mitred path folded into a spike. Cut out of a ring instead,
    # the trace end tapers along the diagonal but stays one clean, DRC-legal shape.
    component = tf_octa_c(
        name=f"tf_test_{next(_names)}", bottom_winding_diameter=24.0, top_winding_diameter=24.0,
        center_displacement=0.0, bottom_linewidth=6.0, top_linewidth=6.0,
        feedline_spacing=8.0, bottom_center_tap_width=3.0, upper_center_tap_width=3.0,
        gnd_upper_spacing=20.0, gnd_lower_spacing=20.0, gnd_side_spacing=20.0, gnd_ring_width=10.0,
    )
    path = str(tmp_path / "wide_gap.gds")
    component.write_gds(path, with_metadata=False)

    assert _metal(path, SG13G2.TopMetal2).count() == 1
    assert check_gds_file(path).clean


def test_ports_inside_the_windings_are_rejected():
    with pytest.raises(ValueError, match="inside the windings"):
        check_tf_octa_c_parameters(top_linewidth=6.0, gnd_upper_spacing=12.0, gnd_ring_width=10.0)


@pytest.mark.parametrize(
    ("change", "feasible"),
    [
        ({"top_winding_diameter": 30.0, "bottom_winding_diameter": 40.0}, True),  # ratio 0.75
        ({"top_winding_diameter": 29.9, "bottom_winding_diameter": 40.0}, False),
        ({"relative_displacement": 0.2}, True),  # the largest offset, at the smallest size
        ({"bottom_linewidth": 6.7}, False),  # wider than a third of the diameter
    ],
)
def test_coupling_and_width_rules(change, feasible):
    assert TransformerOcta().is_feasible(SMALLEST | change) is feasible


def test_layout_carries_its_dimensions_as_text(tmp_path):
    params = SMALLEST | {"bottom_winding_diameter": 40.0, "top_winding_diameter": 30.0}
    path = _draw(tmp_path, params)
    layout = kdb.Layout()
    layout.read(path)

    (text,) = [s.text.string for s in layout.top_cell().shapes(layout.find_layer(*SG13G2.TEXT)).each()]

    assert "bottom winding diameter: 40.00" in text
    assert "top winding diameter: 30.00" in text
    assert "top feed gap: 5.00" in text
    assert check_gds_file(path).clean


def test_offset_scales_with_the_windings_and_lands_on_the_grid(tmp_path):
    params = SMALLEST | {"bottom_winding_diameter": 37.4, "top_winding_diameter": 33.3,
                         "relative_displacement": 0.035}

    # 0.035 of the 35.35 µm mean diameter
    assert TransformerOcta.center_displacement(params) == pytest.approx(1.23725)
    path = _draw(tmp_path, params)
    layout = kdb.Layout()
    layout.read(path)
    (text,) = [s.text.string for s in layout.top_cell().shapes(layout.find_layer(*SG13G2.TEXT)).each()]
    assert "center displacement: 1.240" in text  # each winding centre on the 5 nm grid
    assert check_gds_file(path).snapped_vertices == 0
