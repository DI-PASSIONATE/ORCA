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
