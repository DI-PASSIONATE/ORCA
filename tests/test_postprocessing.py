"""Electrical parameters of the presets' devices, on circuits with known values."""

import math

import numpy as np
import pytest

skrf = pytest.importorskip("skrf")

from orca.geometry.presets import InductorOcta, TransformerOcta
from orca.utils.postprocessing import _first_inductive_to_capacitive, differential_impedance

FREQUENCIES = np.linspace(0.0, 20e9, 81)
L_DIFF = 2e-9
C_SHUNT = 0.5e-12
R_DIFF = 2.0
Z0 = 50.0


def _network(y: np.ndarray, frequencies: np.ndarray) -> "skrf.Network":
    n = y.shape[-1]
    eye = np.eye(n)
    s = (eye - Z0 * y) @ np.linalg.inv(eye + Z0 * y)
    return skrf.Network(frequency=skrf.Frequency.from_f(frequencies, unit="hz"), s=s, z0=Z0)


def _center_tapped_inductor(frequencies=FREQUENCIES) -> "skrf.Network":
    """Two half windings from ports 1 and 2 to the center tap (port 3), with a shunt C at the ends."""
    omega = 2 * np.pi * frequencies
    y_half = 1 / (R_DIFF / 2 + 1j * omega * L_DIFF / 2)
    y_shunt = 1j * omega * C_SHUNT
    y = np.zeros((len(frequencies), 3, 3), dtype=complex)
    y[:, 0, 0] = y[:, 1, 1] = y_half + y_shunt
    y[:, 0, 2] = y[:, 2, 0] = y[:, 1, 2] = y[:, 2, 1] = -y_half
    y[:, 2, 2] = 2 * y_half
    return _network(y, frequencies)


#: Branch inductances [nH] of four coupled half windings: top op->oci, oci->on and bottom
#: ip->ico, ico->in, so a current through either winding runs forward through both halves
HALF_WINDINGS = np.array(
    [
        [0.50, 0.10, 0.15, 0.15],
        [0.10, 0.50, 0.15, 0.15],
        [0.15, 0.15, 0.50, 0.10],
        [0.15, 0.15, 0.10, 0.50],
    ]
)
L_WINDING = 2 * 0.5 + 2 * 0.1  # nH, both halves and their mutual inductance
M_WINDINGS = 4 * 0.15  # nH, every half of one winding with every half of the other


def _transformer(frequency: float = 1e9) -> "skrf.Network":
    """TransformerOcta's six ports: 1 op, 2 on, 3 ip, 4 in, 5 oci, 6 ico."""
    omega = 2 * np.pi * frequency
    z_branch = 0.5 * np.eye(4) + 1j * omega * HALF_WINDINGS * 1e-9
    # Node-branch incidence: each branch leaves one port and enters another
    incidence = np.zeros((6, 4))
    for branch, (start, end) in enumerate([(0, 4), (4, 1), (2, 5), (5, 3)]):
        incidence[start, branch], incidence[end, branch] = 1, -1
    y = incidence @ np.linalg.inv(z_branch) @ incidence.T + 1j * omega * 5e-15 * np.eye(6)
    return _network(y[None], np.array([frequency]))


def test_inductor_preset_reads_the_center_tapped_inductor():
    metrics = InductorOcta().electrical_parameters(_center_tapped_inductor())

    assert set(metrics) == {"L", "R", "Q", "srf_f"}
    low = 1  # 0.25 GHz, well below resonance
    omega = 2 * np.pi * FREQUENCIES[low]
    assert metrics["L"][low] == pytest.approx(L_DIFF * 1e9, rel=1e-2)
    assert metrics["R"][low] == pytest.approx(R_DIFF, rel=1e-2)
    assert metrics["Q"][low] == pytest.approx(omega * L_DIFF / R_DIFF, rel=2e-2)
    # Each half winding resonates with its shunt C: w^2 = 2 / (L C) - (R / L)^2
    resonance = math.sqrt(2 / (L_DIFF * C_SHUNT) - (R_DIFF / L_DIFF) ** 2) / (2 * math.pi)
    assert float(metrics["srf_f"]) == pytest.approx(resonance / 1e9, rel=1e-2)


def test_inductor_below_resonance_has_no_srf():
    metrics = InductorOcta().electrical_parameters(_center_tapped_inductor(FREQUENCIES[:20]))

    assert np.isnan(metrics["srf_f"])


def test_transformer_preset_pairs_the_ports_of_each_winding():
    metrics = TransformerOcta().electrical_parameters(_transformer())

    assert metrics["Lp"][0] == pytest.approx(L_WINDING, rel=1e-2)
    assert metrics["Ls"][0] == pytest.approx(L_WINDING, rel=1e-2)
    assert metrics["Rp"][0] == pytest.approx(1.0, rel=1e-2)
    assert metrics["k"][0] == pytest.approx(M_WINDINGS / L_WINDING, rel=1e-2)
    assert TransformerOcta.absolute_error_parameters == {"k"}


def test_differential_impedance_of_a_floating_source_across_a_series_element():
    # A lone series impedance between two ports: the floating source sees exactly it
    z_series = 3 + 4j
    y = np.array([[1, -1], [-1, 1]]) / z_series
    ntwk = _network(y[None], np.array([1e9]))

    assert differential_impedance(ntwk, [(0, 1)])[0, 0, 0] == pytest.approx(z_series)


def _asymmetric_network() -> "skrf.Network":
    """A passive, reciprocal 4-port without symmetry, so a termination changes the result."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 4))
    b = rng.normal(size=(4, 4))
    y = (a @ a.T + np.eye(4)) * 1e-2 + 1j * (b + b.T) * 1e-2
    return _network(y[None], np.array([1e9]))


def _schur_reference(ntwk, pair, loads: dict[int, complex]) -> complex:
    """The pair's differential impedance with every other port eliminated, loaded by *loads*."""
    y = ntwk.y[0]
    driven = list(pair)
    others = [port for port in range(ntwk.nports) if port not in driven]
    y_load = np.diag([1 / loads[port] if port in loads else 0.0 for port in others])
    y_reduced = y[np.ix_(driven, driven)] - y[np.ix_(driven, others)] @ np.linalg.inv(
        y[np.ix_(others, others)] + y_load
    ) @ y[np.ix_(others, driven)]
    z = np.linalg.inv(y_reduced)
    return z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1]


@pytest.mark.parametrize("load", [50.0, 10 - 20j, 200.0])
def test_a_terminated_port_is_loaded_to_ground(load):
    ntwk = _asymmetric_network()

    z = differential_impedance(ntwk, [(0, 1)], terminations={2: load})[0, 0, 0]

    assert z == pytest.approx(_schur_reference(ntwk, (0, 1), {2: load}))
    # Port 3 stays open; an open port 2 gives another value, so the load mattered.
    assert z != pytest.approx(differential_impedance(ntwk, [(0, 1)])[0, 0, 0], rel=1e-3)


def test_terminations_approach_a_short_and_an_open():
    ntwk = _asymmetric_network()

    shorted = differential_impedance(ntwk, [(0, 1)], shorted=(2,))[0, 0, 0]
    open_ = differential_impedance(ntwk, [(0, 1)])[0, 0, 0]

    def loaded(load: float) -> complex:
        return differential_impedance(ntwk, [(0, 1)], terminations={2: load})[0, 0, 0]

    assert loaded(1e-9) == pytest.approx(shorted)
    assert loaded(1e12) == pytest.approx(open_)


@pytest.mark.parametrize(
    ("shorted", "terminations", "match"),
    [
        ((), {0: 50.0}, "driven pair"),
        ((2,), {2: 50.0}, "both shorted and terminated"),
        ((), {2: 0}, "list the port in shorted"),
    ],
)
def test_contradictory_terminations_are_rejected(shorted, terminations, match):
    with pytest.raises(ValueError, match=match):
        differential_impedance(_asymmetric_network(), [(0, 1)], shorted, terminations)


def test_transformer_parameters_take_center_tap_terminations():
    from orca.utils.postprocessing import transformer_parameters

    ntwk = _transformer()
    loaded = transformer_parameters(ntwk, (0, 1), (2, 3), terminations={4: 50.0, 5: 50.0})
    grounded = transformer_parameters(ntwk, (0, 1), (2, 3), shorted=(4, 5))

    # The windings are symmetric, so their center taps carry no differential current.
    assert loaded["Lp"][0] == pytest.approx(grounded["Lp"][0])
    assert loaded["k"][0] == pytest.approx(grounded["k"][0])


def test_tester_reports_the_geometry_errors():
    from orca.pipeline.test_model_stage import ModelTester

    reference = _center_tapped_inductor()
    predicted = reference.copy()
    predicted.s = predicted.s * 0.99

    medians, curves = ModelTester._electrical_errors(
        predicted, reference, "inductor", InductorOcta()
    )

    assert set(medians) == {"L error %", "R error %", "Q error %", "srf_f error %"}
    assert set(curves) == {"L", "R", "Q"}
    assert all(np.isfinite(error) and error > 0 for error in medians.values())


def test_srf_ignores_the_sign_of_the_reactance_at_dc():
    # An open winding is capacitive from DC on; rounding can leave +0 reactance at DC
    frequencies = np.array([0.0, 1e9, 2e9])
    reactance = np.array([1e-12, -500.0, -250.0])

    assert np.isnan(_first_inductive_to_capacitive(frequencies / 1e9, reactance))


def test_srf_ignores_sign_noise_just_above_dc():
    # Reactance of TransformerOcta sample 2598 as simulated: milliohm noise flips sign
    # at 20 MHz, long before the real resonance between 396 and 398 GHz
    frequencies = np.array([0.0, 0.01, 0.02, 0.1, 0.2, 390.0, 392.0, 394.0, 396.0, 398.0, 400.0])
    reactance = np.array([2e-15, 0.0068, -0.019, 0.0017, 0.020, 388, 373, 299, 152, -15.9, -127])

    assert _first_inductive_to_capacitive(frequencies, reactance) == pytest.approx(397.8, abs=0.1)
