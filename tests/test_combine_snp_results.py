"""DC extrapolation of the merged Palace results."""

import os

import numpy as np
import pytest

skrf = pytest.importorskip("skrf")

from orca.simulation.combine_snp_results import extrapolate_to_DC

R_SERIES = 2.0
L_SERIES = 1e-9


def _series_rl(frequencies: np.ndarray) -> "skrf.Network":
    """A series R-L between two 50 Ohm ports."""
    z = R_SERIES + 2j * np.pi * frequencies * L_SERIES
    s11 = z / (z + 100)
    s21 = 100 / (z + 100)
    s = np.stack([np.stack([s11, s21], -1), np.stack([s21, s11], -1)], -2)
    return skrf.Network(frequency=skrf.Frequency.from_f(frequencies, unit="hz"), s=s, z0=50)


def test_uneven_sweep_keeps_its_frequencies(tmp_path):
    # Points near DC added to a 1 GHz sweep, as the presets' fpoint does
    frequencies = np.concatenate(([10e6, 20e6, 0.1e9, 0.5e9], np.arange(1, 31) * 1e9))
    path = os.path.join(tmp_path, "rl.s2p")
    _series_rl(frequencies).write_touchstone(path)

    result = skrf.Network(extrapolate_to_DC(path) + ".s2p")

    np.testing.assert_allclose(result.f, np.concatenate(([0.0], frequencies)))
    np.testing.assert_allclose(result.s[1:], _series_rl(frequencies).s, atol=1e-9)
    # From 10 and 20 MHz, the DC point is the series resistance
    np.testing.assert_allclose(result.s[0], _series_rl(np.array([0.0])).s[0].real, atol=1e-4)


@pytest.mark.parametrize("n_ports", [2, 3])
def test_port_deembedding_removes_each_ports_own_inductance(tmp_path, n_ports):
    """A series L at port k only adds jwL to Z_kk, so de-embedding must take exactly that off."""
    from orca.simulation.combine_snp_results import flat_strip_inductance, port_deembedding

    rng = np.random.default_rng(0)
    frequencies = np.linspace(1e9, 30e9, 30)
    s = 0.4 * (rng.standard_normal((30, n_ports, n_ports)) + 1j * rng.standard_normal((30, n_ports, n_ports)))
    s = (s + s.transpose(0, 2, 1)) / 2  # reciprocal, like a passive layout
    raw = skrf.Network(frequency=skrf.Frequency.from_f(frequencies, unit="hz"), s=s, z0=50)
    path = os.path.join(tmp_path, f"dut.s{n_ports}p")
    raw.write_touchstone(path)

    # Listed out of port order, with different sizes, as port_information.json may hold them
    sizes = {1: (5.0, 2.0), 2: (8.0, 3.0), 3: (6.0, 1.5)}
    ports = [
        {"portnumber": number, "length": length, "width": width}
        for number, (length, width) in reversed(list(sizes.items())[:n_ports])
    ]
    port_deembedding(path, True, {"ports": ports})

    result = skrf.Network(os.path.join(tmp_path, f"dut_deembedded.s{n_ports}p"))
    inductances = [flat_strip_inductance(*sizes[k + 1], 0, 1e-6) for k in range(n_ports)]
    omega = 2 * np.pi * frequencies[:, None, None]
    expected_z = raw.z - 1j * omega * np.diag(inductances)
    np.testing.assert_allclose(result.z, expected_z, rtol=1e-6, atol=1e-6)
