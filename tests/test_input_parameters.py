"""Tests for the geometry parameter objects and the iterator that samples them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orca.geometry.input_parameters import (
    ChoiceParameter,
    InputParameterIterator,
    RangeParameter,
)


def test_step_grid_includes_both_bounds_and_rounds_to_the_step():
    width = RangeParameter("width", 2.0, 15.0, step=0.02)

    values = width.values_list()

    assert values[0] == 2.0
    assert values[-1] == 15.0
    assert len(values) == 651
    assert RangeParameter("x", 0.0, 1.0, step=0.1).values_list()[3] == 0.3


def test_step_must_divide_the_range():
    with pytest.raises(ValueError, match="does not divide"):
        RangeParameter("x", 0.0, 1.0, step=0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min": 2.0, "max": 1.0}, "below min"),
        ({"min": 0.0, "max": 1.0, "sampling": "log"}, "min > 0"),
        ({"min": 0.0, "max": 1.0, "sampling": "normal"}, "unknown sampling"),
        ({"min": 0.5, "max": 3.0, "dtype": int}, "integer min, max and step"),
    ],
)
def test_invalid_ranges_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RangeParameter("x", **kwargs)


def test_integer_parameter_steps_by_one_and_yields_ints():
    turns = RangeParameter("turns", 1, 5, dtype=int)

    assert turns.values_list() == [1, 2, 3, 4, 5]
    assert all(type(turns.from_unit(q)) is int for q in (0.0, 0.37, 1.0))


def test_quantiles_map_to_the_bounds():
    for parameter in (
        RangeParameter("a", 30.0, 300.0, step=2.0, sampling="log"),
        RangeParameter("b", 30.0, 300.0, sampling="log"),
        RangeParameter("c", 0.0, 15.0),
        ChoiceParameter("d", [5, 1, 3]),
    ):
        low, high = parameter.bounds
        assert parameter.from_unit(0.0) == pytest.approx(low)
        assert parameter.from_unit(1.0) == pytest.approx(high)


def test_log_sampling_draws_half_the_values_below_the_geometric_mean():
    rng = np.random.default_rng(0)
    for step in (2.0, None):
        diameter = RangeParameter("diameter", 30.0, 300.0, step=step, sampling="log")
        draws = np.array([diameter.from_unit(q) for q in rng.random(20_000)])

        assert np.mean(draws < math.sqrt(30.0 * 300.0)) == pytest.approx(0.5, abs=0.02)


def test_uniform_grid_draws_are_equally_likely():
    space = RangeParameter("space", 2.0, 2.2, step=0.1)
    draws = [space.from_unit(q) for q in np.random.default_rng(0).random(30_000)]

    shares = [draws.count(value) / len(draws) for value in space.values_list()]

    assert shares == pytest.approx([1 / 3] * 3, abs=0.01)


def test_choice_yields_only_its_values():
    choice = ChoiceParameter("layer", [134, 126, 50])
    draws = {choice.from_unit(q) for q in np.random.default_rng(0).random(1000)}

    assert draws == {50, 126, 134}
    assert all(type(value) is int for value in draws)


def test_random_draws_stay_on_the_grid_and_repeat_with_the_seed():
    def draws(seed: int) -> list[dict]:
        iterator = InputParameterIterator(
            RangeParameter("turns", 1, 5, dtype=int),
            RangeParameter("diameter", 30.0, 300.0, step=2.0, sampling="log"),
        )
        iterator.set_sample_count(200, seed=seed)
        return list(iterator)

    first = draws(seed=3)

    assert first == draws(seed=3)
    assert first != draws(seed=4)
    diameters = set(RangeParameter("d", 30.0, 300.0, step=2.0).values_list())
    assert all(sample["diameter"] in diameters for sample in first)


def test_grid_reaches_every_parameters_bounds():
    iterator = InputParameterIterator(
        RangeParameter("a", 0.0, 9.0, step=1.0),
        RangeParameter("b", 1.0, 100.0, sampling="log"),
        picking_strategy="grid",
    )
    iterator.set_sample_count(9)

    samples = list(iterator)

    a_values = sorted({s["a"] for s in samples})
    assert len(a_values) == 3
    assert (a_values[0], a_values[-1]) == (0.0, 9.0)
    assert sorted({s["b"] for s in samples}) == pytest.approx([1.0, 10.0, 100.0])


def test_step_grid_needs_discrete_parameters():
    iterator = InputParameterIterator(
        RangeParameter("a", 0.0, 1.0), picking_strategy="step_grid"
    )

    with pytest.raises(ValueError, match="continuous"):
        iterator.set_sample_count(10)


def test_iterator_rejects_bad_declarations():
    with pytest.raises(ValueError, match="repeated"):
        InputParameterIterator(RangeParameter("a", 0, 1), RangeParameter("a", 0, 2))
    with pytest.raises(ValueError, match="reserved"):
        InputParameterIterator(RangeParameter("frequency", 1.0, 2.0))
    with pytest.raises(TypeError, match="RangeParameter or ChoiceParameter"):
        InputParameterIterator([1, 2, 3])  # ty: ignore[invalid-argument-type]


def test_ranges_list_parameters_then_frequency():
    iterator = InputParameterIterator(
        RangeParameter("turns", 1, 5, dtype=int),
        ChoiceParameter("layer", [134, 126]),
        frequency=[1e9, 500e9],
    )

    assert iterator.get_ranges() == {
        "turns": {"min": 1.0, "max": 5.0},
        "layer": {"min": 126.0, "max": 134.0},
        "frequency": {"min": 1e9, "max": 500e9},
    }
    assert iterator.get_min_max_values() == ([1.0, 126.0, 1e9], [5.0, 134.0, 500e9])


def _unit_box(strategy: str, n: int, seed: int, boundary_fraction: float = 0.0) -> np.ndarray:
    iterator = InputParameterIterator(
        *(RangeParameter(name, 0.0, 1.0) for name in "abcd"),
        picking_strategy=strategy,
        boundary_fraction=boundary_fraction,
    )
    iterator.set_sample_count(n, seed=seed)
    return np.array([list(sample.values()) for sample in iterator])


@pytest.mark.parametrize("strategy", ["sobol", "lhs"])
def test_space_filling_designs_cover_the_box_more_evenly_than_random_draws(strategy):
    from scipy.stats import qmc

    random_points = _unit_box("random", 512, seed=1)
    filled = _unit_box(strategy, 512, seed=1)

    assert qmc.discrepancy(filled) < qmc.discrepancy(random_points) / 2


@pytest.mark.parametrize("strategy", ["sobol", "lhs", "random"])
def test_drawing_strategies_repeat_with_the_seed_and_replace_rejections(strategy):
    def draws(seed: int) -> tuple[list[dict], int]:
        iterator = InputParameterIterator(
            RangeParameter("a", 0.0, 1.0),
            RangeParameter("b", 0.0, 1.0, step=0.01),
            picking_strategy=strategy,
        )
        iterator.set_sample_count(300, seed=seed, feasible=lambda s: s["a"] + s["b"] <= 1)
        return list(iterator), iterator.n_rejected

    (first, rejected), (again, _) = draws(5), draws(5)

    assert first == again
    assert len(first) == 300
    assert rejected > 0
    assert all(s["a"] + s["b"] <= 1 for s in first)


def test_rejected_draws_are_recorded():
    iterator = InputParameterIterator(RangeParameter("a", 0.0, 1.0))
    iterator.set_sample_count(100, seed=0, feasible=lambda s: s["a"] < 0.5)

    accepted = list(iterator)

    assert len(iterator.rejected) == iterator.n_rejected > 0
    assert all(s["a"] >= 0.5 for s in iterator.rejected)
    assert all(s["a"] < 0.5 for s in accepted)


def test_boundary_fraction_moves_draws_onto_the_faces_of_the_box():
    def on_boundary(points: np.ndarray) -> np.ndarray:
        return ((points == 0.0) | (points == 1.0)).any(axis=1)

    assert not on_boundary(_unit_box("sobol", 256, seed=2)).any()
    assert on_boundary(_unit_box("sobol", 256, seed=2, boundary_fraction=1.0)).all()
    share = on_boundary(_unit_box("sobol", 4096, seed=2, boundary_fraction=0.1)).mean()
    assert share == pytest.approx(0.1, abs=0.02)
    # Corners too, not only faces: every coordinate pinned in some draws
    corners = ((_unit_box("random", 4096, seed=2, boundary_fraction=1.0) % 1 == 0).all(axis=1))
    assert corners.any()


def test_boundary_fraction_must_be_a_share():
    with pytest.raises(ValueError, match="boundary_fraction"):
        InputParameterIterator(RangeParameter("a", 0.0, 1.0), boundary_fraction=1.5)


def _triangle(kept: tuple[str, ...], seed: int = 4) -> tuple[np.ndarray, InputParameterIterator]:
    """Draws from the unit square under a + b <= 1, which plain rejection skews towards small a."""
    iterator = InputParameterIterator(
        RangeParameter("a", 0.0, 1.0), RangeParameter("b", 0.0, 1.0),
        boundary_fraction=0.0, kept_on_rejection=kept,
    )
    iterator.set_sample_count(4000, seed=seed, feasible=lambda s: s["a"] + s["b"] <= 1)
    return np.array([s["a"] for s in iterator]), iterator


def test_kept_parameters_keep_their_distribution_through_rejections():
    plain, _ = _triangle(kept=())
    kept, iterator = _triangle(kept=("a",))

    # Under plain rejection a follows the triangle's density 2 * (1 - a): mean 1/3
    assert plain.mean() == pytest.approx(1 / 3, abs=0.02)
    # Kept, it stays uniform; only points too close to 1 for any b to fit are lost
    assert kept.mean() == pytest.approx(0.5, abs=0.02)
    assert len(kept) == 4000
    assert iterator.n_rejected > 0


def test_redraws_repeat_with_the_seed():
    first, _ = _triangle(kept=("a",), seed=8)
    again, _ = _triangle(kept=("a",), seed=8)

    assert np.array_equal(first, again)


def test_kept_parameters_must_exist():
    with pytest.raises(ValueError, match="unknown parameters"):
        InputParameterIterator(RangeParameter("a", 0.0, 1.0), kept_on_rejection=("diameter",))
