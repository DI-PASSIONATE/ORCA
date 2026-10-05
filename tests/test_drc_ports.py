"""Tests for the port contact check: every gds2palace port marker must touch its metals."""

from __future__ import annotations

import klayout.db as kdb
import pandas as pd
import pytest

from orca.geometry.drc import PortContact, check_gds_file, check_ports, port_contacts
from orca.geometry.layers import SG13G2
from orca.geometry.presets import InductorOcta, StackupXML, TransformerOcta
from orca.pipeline.context import PipelineContext
from orca.pipeline.drc_stage import DRCChecker
from orca.simulation.simulate import read_simconfig

#: A vertical port from Metal5 up to TopMetal2, marked on layer 201
PORT = PortContact(number=1, marker_layer=201, metals=(("Metal5", 67), ("TopMetal2", 134)))


def _layout(marker_x_um: float, marker: str = "path") -> kdb.Layout:
    """A TopMetal2 feed ending at x = 0 over a Metal5 bar starting there, plus a port marker."""
    layout = kdb.Layout()
    top = layout.create_cell("top")
    um = lambda value: round(value / layout.dbu)  # noqa: E731 - one-line unit helper
    top.shapes(layout.layer(*SG13G2.TopMetal2)).insert(kdb.Box(um(-20), um(-2), 0, um(2)))
    top.shapes(layout.layer(*SG13G2.Metal5)).insert(kdb.Box(0, um(-10), um(10), um(10)))
    x = um(marker_x_um)
    if marker == "path":
        line = kdb.Path([kdb.Point(x, um(-2)), kdb.Point(x, um(2))], 0)
        top.shapes(layout.layer(201, 0)).insert(line)
    else:
        top.shapes(layout.layer(201, 0)).insert(kdb.Box(x - um(0.1), um(-2), x, um(2)))
    return layout


@pytest.mark.parametrize("marker", ["path", "box"])
def test_marker_on_the_feed_end_is_connected(marker):
    assert check_ports(_layout(0.0, marker), (PORT,)) == {}


@pytest.mark.parametrize("marker", ["path", "box"])
def test_marker_5nm_off_the_feed_end_is_open(marker):
    # Beyond the feed end, over the ground bar: it touches Metal5 but not the feed
    assert check_ports(_layout(0.105, marker), (PORT,)) == {"port1.open_TopMetal2": 1}


def test_missing_marker_is_reported():
    layout = _layout(0.0)
    layout.clear_layer(layout.find_layer(201, 0))

    assert check_ports(layout, (PORT,)) == {"port1.missing": 1}


def test_port_contacts_follow_the_simconfig_and_stackup():
    geometry = TransformerOcta()
    contacts = port_contacts(read_simconfig(geometry.simconfig_filename), geometry.stackup_xml)

    assert [c.number for c in contacts] == [1, 2, 3, 4, 5, 6]
    assert contacts[0] == PORT
    assert contacts[2].metals == (("Metal5", 67), ("TopMetal1", 126))


def test_port_on_a_layer_the_stackup_lacks_is_rejected():
    simconfig = {"ports": [{"portnumber": 1, "source_layernum": 201,
                            "from_layername": "Metal5", "to_layername": "Metal9"}]}

    with pytest.raises(ValueError, match="Metal9"):
        port_contacts(simconfig, StackupXML.SG13G2_FEM_200um)


@pytest.mark.parametrize("geometry", [InductorOcta(), TransformerOcta()], ids=lambda g: g.name)
def test_sampled_preset_layouts_have_every_port_connected(tmp_path, geometry):
    ports = port_contacts(read_simconfig(geometry.simconfig_filename), geometry.stackup_xml)
    iterator = geometry.input_parameter_iterator
    iterator.set_sample_count(30, seed=5, feasible=geometry.is_feasible)

    failures = {}
    for i, params in enumerate(iterator):
        path = str(tmp_path / f"{geometry.name}_{i}.gds")
        geometry.create_gds_file(f"{geometry.name}_{i}", path, params)
        result = check_gds_file(path, ports=ports)
        if not result.ports_connected:
            failures[str(params)] = result.port_findings

    assert not failures


def test_drc_stage_drops_open_ports_even_when_violations_are_kept(tmp_path):
    context = PipelineContext(geometry=TransformerOcta(), base_dir=str(tmp_path), num_processes=1)
    gds_dir = context.geometry_dir
    tmp_path.joinpath(gds_dir).mkdir(parents=True, exist_ok=True)
    good = {"bottom_winding_diameter": 21.4, "top_winding_diameter": 20.1,
            "relative_displacement": 0.135, "bottom_linewidth": 7.0, "top_linewidth": 5.2}
    rows = []
    for name, x in (("connected.gds", 0.0), ("open.gds", 0.105)):
        if name == "connected.gds":
            TransformerOcta.create_gds_file("connected", f"{gds_dir}/{name}", good)
        else:
            _layout(x).write(f"{gds_dir}/{name}")
        rows.append({"name": name} | good)
    pd.DataFrame(rows).to_csv(context.gds_csv_path, index=False)

    context = DRCChecker(drop_violations=False).run(context)

    passed = pd.read_csv(context.drc_csv_path)
    assert list(passed["name"]) == ["connected.gds"]
    report = pd.read_csv(context.drc_report_path).set_index("name")
    assert "port1.open_TopMetal2:1" in report.loc["open.gds", "rules"]
