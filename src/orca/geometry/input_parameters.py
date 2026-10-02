"""Geometry input parameters and the iterator that samples them for GDS generation.

A geometry declares each of its parameters as an object - :class:`RangeParameter` for
a numeric range, :class:`ChoiceParameter` for an explicit list of values - and hands
them to an :class:`InputParameterIterator`, which draws the combinations to lay out.

Every parameter maps a number ``q`` in [0, 1] to one of its values through its
sampling distribution (:meth:`GeometryParameter.from_unit`). The space-filling
strategies feed it Sobol' or Latin hypercube points, the random strategy uniform
random numbers and the grid strategies evenly spaced ones, so a parameter sampled on
a log scale is also gridded and space-filled on a log scale.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: Draws the 'random' strategy may spend per requested sample before giving up
#: on a constraint that rejects almost everything.
MAX_DRAWS_PER_SAMPLE = 100

#: How a parameter's values are spread over its range: evenly (``"uniform"``), or
#: evenly in the logarithm (``"log"``), which draws as many values from 30 to 100 as
#: from 100 to 300 and so favours the small end of the range.
Sampling = Literal["uniform", "log"]
SAMPLINGS: tuple[Sampling, ...] = ("uniform", "log")

#: Picking strategies of :class:`InputParameterIterator`.
PICKING_STRATEGIES = ("sobol", "lhs", "random", "grid", "uniform_grid", "step_grid")

#: Strategies that draw points until enough feasible ones were accepted.
DRAWING_STRATEGIES = ("sobol", "lhs", "random")

#: Rejected draws kept in :attr:`InputParameterIterator.rejected` for plotting.
MAX_RECORDED_REJECTIONS = 20_000


class GeometryParameter(ABC):
    """One input parameter of a geometry: its name, its values and how they are drawn."""

    name: str

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Smallest and largest value the parameter can take."""

    @property
    @abstractmethod
    def grid_values(self) -> np.ndarray | None:
        """Every value the parameter can take, ascending, or None if it is continuous."""

    @abstractmethod
    def values_list(self) -> list[int | float]:
        """Every value the parameter can take, ascending, as Python numbers.

        Raises:
            ValueError: If the parameter is continuous.
        """

    @abstractmethod
    def from_unit(self, q: float) -> int | float:
        """The value at quantile ``q`` in [0, 1] of the sampling distribution.

        ``q = 0`` gives the smallest and ``q = 1`` the largest value; a uniform random
        ``q`` gives a random draw from the distribution.
        """


def _decimals(value: float) -> int:
    """Number of decimal places ``value`` is written with, e.g. 2 for 0.02 and 0 for 30.0."""
    exponent = Decimal(repr(float(value))).normalize().as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _discrete_quantile(values: np.ndarray, cdf: np.ndarray, q: float) -> Any:
    """The value whose share of the cumulative weights ``cdf`` contains ``q``."""
    index = int(np.searchsorted(cdf, q, side="right"))
    return values[min(index, len(values) - 1)]


@dataclass(frozen=True)
class RangeParameter(GeometryParameter):
    """A numeric parameter between ``min`` and ``max``, both inclusive.

    Args:
        name (str): Parameter name, as passed to ``create_gds_file`` and stored in the
            parameter table and the ONNX metadata.
        min (float): Smallest value.
        max (float): Largest value. Equal to ``min`` for a fixed parameter.
        step (float | None): Spacing of the allowed values, which are ``min``,
            ``min + step``, ... up to ``max``; it must divide ``max - min``. Values are
            rounded to the decimals of ``min`` and ``step``, so ``step=0.1`` yields 0.3
            rather than 0.30000000000000004. ``None`` makes the parameter continuous;
            integer parameters default to a step of 1.
        sampling (Sampling): ``"uniform"`` spreads draws evenly over the range;
            ``"log"`` spreads them evenly in the logarithm, favouring small values
            (needs ``min > 0``).
        dtype (type[int | float]): Type of the values handed to the geometry.
    """

    name: str
    min: float
    max: float
    step: float | None = None
    sampling: Sampling = "uniform"
    dtype: type[int | float] = float

    def __post_init__(self) -> None:
        if self.max < self.min:
            raise ValueError(f"{self.name}: max={self.max} is below min={self.min}.")
        if self.sampling not in SAMPLINGS:
            raise ValueError(
                f"{self.name}: unknown sampling {self.sampling!r}; choose one of {list(SAMPLINGS)}."
            )
        if self.sampling == "log" and self.min <= 0:
            raise ValueError(f"{self.name}: log sampling needs min > 0, got min={self.min}.")
        if self.dtype not in (int, float):
            raise TypeError(f"{self.name}: dtype must be int or float, got {self.dtype!r}.")
        if self.dtype is int:
            step = 1 if self.step is None else self.step
            object.__setattr__(self, "step", step)  # frozen dataclass
            if any(float(v) != int(v) for v in (self.min, self.max, step)):
                raise ValueError(
                    f"{self.name}: an integer parameter needs integer min, max and step, got "
                    f"min={self.min}, max={self.max}, step={self.step}."
                )
        if self.step is not None:
            if self.step <= 0:
                raise ValueError(f"{self.name}: step must be positive, got {self.step}.")
            n_steps = (self.max - self.min) / self.step
            if abs(n_steps - round(n_steps)) > 1e-9 * max(1.0, n_steps):
                raise ValueError(
                    f"{self.name}: step={self.step} does not divide max - min = "
                    f"{self.max - self.min:g}, so max={self.max} would not be a value."
                )

    @property
    def bounds(self) -> tuple[float, float]:
        return float(self.min), float(self.max)

    @cached_property
    def grid_values(self) -> np.ndarray | None:
        if self.step is None:
            return None
        n_steps = round((self.max - self.min) / self.step)
        values = self.min + np.arange(n_steps + 1) * self.step
        return np.round(values, max(_decimals(self.min), _decimals(self.step)))

    @cached_property
    def _cdf(self) -> np.ndarray:
        """Cumulative weights of the grid values under the sampling distribution."""
        values = self.grid_values
        if values is None:  # continuous: from_unit never asks
            return np.empty(0)
        # A log-uniform density is proportional to 1/x
        weights = np.ones(len(values)) if self.sampling == "uniform" else 1.0 / values
        return np.cumsum(weights) / weights.sum()

    def values_list(self) -> list[int | float]:
        values = self.grid_values
        if values is None:
            raise ValueError(f"{self.name} is continuous; it has no finite list of values.")
        return [self.dtype(v) for v in values.tolist()]

    def from_unit(self, q: float) -> int | float:
        q = min(max(float(q), 0.0), 1.0)
        values = self.grid_values
        if values is not None:
            value = _discrete_quantile(values, self._cdf, q)
        elif self.sampling == "log":
            value = math.exp(math.log(self.min) + q * (math.log(self.max) - math.log(self.min)))
        else:
            value = self.min + q * (self.max - self.min)
        return self.dtype(value)


@dataclass(frozen=True)
class ChoiceParameter(GeometryParameter):
    """A parameter that takes one of an explicit list of values, each equally likely.

    Args:
        name (str): Parameter name.
        values (Sequence[float]): The allowed values. Integers stay integers.
    """

    name: str
    values: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            raise ValueError(f"{self.name}: a choice needs at least one value.")
        object.__setattr__(self, "values", tuple(sorted(self.values)))  # frozen dataclass

    @property
    def bounds(self) -> tuple[float, float]:
        return float(min(self.values)), float(max(self.values))

    @property
    def grid_values(self) -> np.ndarray:
        return np.array(self.values)

    @cached_property
    def _cdf(self) -> np.ndarray:
        return np.arange(1, len(self.values) + 1) / len(self.values)

    def values_list(self) -> list[int | float]:
        return list(self.values)

    def from_unit(self, q: float) -> int | float:
        value = _discrete_quantile(np.array(self.values, dtype=object), self._cdf, q)
        return value.item() if isinstance(value, np.generic) else value


class InputParameterIterator:
    """
    The input parameters of a geometry, and the iterator that draws their combinations.

    Example::

        InputParameterIterator(
            RangeParameter("turns", 1, 5, dtype=int),
            RangeParameter("diameter", 30, 300, step=2, sampling="log"),
            picking_strategy="sobol",
            frequency=[1e9, 500e9],
        )

    Attributes:
        parameters (dict[str, GeometryParameter]): The parameters by name, in order.
        input_names (list[str]): Parameter names, in the column order of the parameter table.
        n_rejected (int): Draws the feasibility check rejected since the last
            :meth:`set_sample_count`.
        rejected (list[dict[str, Any]]): Those rejected draws, up to
            :data:`MAX_RECORDED_REJECTIONS` of them; they show where the parameter box
            cannot be built.
    """

    def __init__(
        self,
        *parameters: GeometryParameter,
        picking_strategy: str = "sobol",
        frequency: Sequence[float] | None = None,
        boundary_fraction: float = 0.05,
    ):
        """
        Args:
            *parameters (GeometryParameter): The geometry's parameters, in the order of
                the parameter table and the model inputs.
            picking_strategy (str): How combinations are drawn.

                - ``"sobol"`` (default): a scrambled Sobol' sequence, a space-filling
                  design that covers the parameter box more evenly than random draws, so
                  fewer samples leave fewer gaps.
                - ``"lhs"``: Latin hypercube designs of ``n_samples`` points each; every
                  parameter's range is split into ``n_samples`` strata, one point each.
                - ``"random"``: independent random draws.
                - ``"grid"`` (alias ``"uniform_grid"``): about
                  ``n_samples ** (1 / n_parameters)`` quantiles of each parameter, combined.
                - ``"step_grid"``: every value of every parameter combined (all must have
                  a step or be choices).

                All but the grids draw each parameter through its sampling distribution.
            frequency (Sequence[float] | None): Frequency band of the model, as its lowest
                and highest frequency in Hz. Not drawn (Palace sweeps it), but part of the
                input ranges used for normalisation and the ONNX metadata.
            boundary_fraction (float): Share of the drawn combinations (with the
                ``"sobol"``, ``"lhs"`` and ``"random"`` strategies) that are moved onto the
                boundary of the parameter box: a random number of their parameters, at
                least one, is set to its min or max. Space-filling and random draws almost
                never reach the faces, edges and corners of the box, yet a network
                extrapolates worst there and an optimizer often ends there. 0 disables it.
        """
        for parameter in parameters:
            if not isinstance(parameter, GeometryParameter):
                raise TypeError(
                    f"Parameters must be RangeParameter or ChoiceParameter objects, got "
                    f"{parameter!r}."
                )
        names = [parameter.name for parameter in parameters]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Parameter names must be unique; repeated: {duplicates}.")
        if "frequency" in names:
            raise ValueError("'frequency' is reserved for the frequency band; pass it as frequency=.")
        if picking_strategy not in PICKING_STRATEGIES:
            raise ValueError(
                f"Unknown picking_strategy {picking_strategy!r}; choose one of "
                f"{list(PICKING_STRATEGIES)}."
            )
        if not 0 <= boundary_fraction <= 1:
            raise ValueError(f"boundary_fraction must lie in [0, 1], got {boundary_fraction}.")

        self.parameters: dict[str, GeometryParameter] = {p.name: p for p in parameters}
        self.picking_strategy = picking_strategy
        self.frequency = frequency
        self.boundary_fraction = boundary_fraction
        # Set with the sample count; GDSGenerator passes the seed of the run
        self.seed: int | None = None
        self.n_inputs = len(parameters)
        self.input_names = names

        self._lock = threading.Lock()  # For thread-safe iteration

        # Created after set_sample_count is called
        self._iterator: Iterator[list[Any]] | None = None
        self.feasible: Callable[[dict[str, Any]], bool] | None = None
        self.n_samples = 0
        self.n_rejected = 0
        self.rejected: list[dict[str, Any]] = []
        self.n_accepted = 0
        self.n_geometries_created = 0

    def set_sample_count(
        self,
        n_samples: int,
        seed: int | None = None,
        feasible: Callable[[dict[str, Any]], bool] | None = None,
    ):
        """
        Sets the number of samples to generate and restarts the iteration.

        Args:
            n_samples (int): Number of samples to generate.
            seed (int|None): Seed of the drawing strategies. None draws fresh entropy.
            feasible (Callable|None): Constraint between parameters, typically a geometry's
                ``is_feasible``. Combinations it rejects are skipped and counted in
                :attr:`n_rejected`. The drawing strategies ('sobol', 'lhs', 'random') draw
                on until ``n_samples`` feasible combinations were produced (up to
                :data:`MAX_DRAWS_PER_SAMPLE` draws per sample); the grid strategies
                simply yield fewer points.
        """
        self.n_samples = n_samples
        self.feasible = feasible
        self.n_rejected = 0
        self.rejected = []
        self.n_accepted = 0
        self.seed = seed
        if self.picking_strategy == "step_grid":
            self._iterator = self.step_grid()
        elif self.picking_strategy in ("uniform_grid", "grid"):
            self._iterator = self.uniform_grid()
        else:
            self._iterator = self.drawn_samples()

    def __len__(self):
        """
        THIS RETURNS THE AMOUNT OF INPUT PARAMETERS, NOT THE AMOUNT OF GEOMETRIES THAT CAN BE GENERATED!
        """
        return self.n_inputs

    def __iter__(self):
        self.n_geometries_created = 0
        return self

    def __next__(self) -> dict[str, Any]:
        with self._lock:  # Ensure thread-safe access
            if self._iterator is None:
                raise RuntimeError(
                    "set_sample_count() must be called before iterating over the input parameters."
                )
            while True:
                if (
                    self.picking_strategy in DRAWING_STRATEGIES
                    and self.n_accepted >= self.n_samples
                ):
                    raise StopIteration
                # Raises StopIteration when exhausted, which is propagated to this iterator
                params = next(self._iterator)
                # Python scalars rather than numpy ones, for the libraries downstream
                params = [
                    param.item() if isinstance(param, np.generic) else param for param in params
                ]
                sample = dict(zip(self.input_names, params, strict=True))
                if self.feasible is not None and not self.feasible(sample):
                    self.n_rejected += 1
                    if len(self.rejected) < MAX_RECORDED_REJECTIONS:
                        self.rejected.append(sample)
                    continue
                self.n_accepted += 1
                self.n_geometries_created += 1  # May be used for logging or tracking
                return sample

    def get_min_max_values(self) -> tuple[list[float], list[float]]:
        """
        Returns the minimum and maximum of each input parameter, then of the frequency.
        Useful for normalization purposes.

        Returns:
            tuple: A tuple containing two lists - (min_values, max_values).
        """
        ranges = self.get_ranges()
        return (
            [r["min"] for r in ranges.values()],
            [r["max"] for r in ranges.values()],
        )

    def get_ranges(self) -> dict[str, dict[str, float]]:
        """
        Returns a dictionary mapping parameter names, then ``frequency``, to their
        ``{"min": ..., "max": ...}`` range.
        """
        ranges = {}
        for name, parameter in self.parameters.items():
            low, high = parameter.bounds
            ranges[name] = {"min": low, "max": high}
        if self.frequency is not None:
            ranges["frequency"] = {
                "min": float(min(self.frequency)),
                "max": float(max(self.frequency)),
            }
        return ranges

    def step_grid(self) -> Iterator[list[Any]]:
        """
        Every combination of every value of every parameter.

        Raises:
            ValueError: If a parameter is continuous (a RangeParameter without a step).
        """
        continuous = [p.name for p in self.parameters.values() if p.grid_values is None]
        if continuous:
            raise ValueError(
                f"The 'step_grid' strategy needs a step on every parameter; {continuous} are "
                "continuous. Give them a step, or use the 'grid' strategy."
            )
        value_lists = [parameter.values_list() for parameter in self.parameters.values()]
        return (list(combination) for combination in product(*value_lists))

    def uniform_grid(self) -> Iterator[list[Any]]:
        """
        A grid of about ``n_samples`` points: evenly spaced quantiles of each parameter's
        sampling distribution, combined. Each parameter's grid includes its min and max.
        """
        steps_per_param = math.ceil(self.n_samples ** (1 / self.n_inputs)) if self.n_inputs else 1
        quantiles = np.linspace(0.0, 1.0, steps_per_param) if steps_per_param > 1 else [0.0]
        value_lists = []
        for parameter in self.parameters.values():
            values: list[Any] = []
            for q in quantiles:
                value = parameter.from_unit(float(q))
                if value not in values:  # a short list of values repeats at nearby quantiles
                    values.append(value)
            value_lists.append(values)
        return (list(combination) for combination in product(*value_lists))

    def drawn_samples(self) -> Iterator[list[Any]]:
        """
        Combinations drawn with the 'sobol', 'lhs' or 'random' strategy, a
        ``boundary_fraction`` of them moved onto the boundary of the parameter box.
        """
        # Independent streams for the points and for the boundary moves, both from the seed
        point_seed, boundary_seed = np.random.SeedSequence(self.seed).spawn(2)
        points = self._unit_points(np.random.default_rng(point_seed))
        boundary_rng = np.random.default_rng(boundary_seed)
        parameters = list(self.parameters.values())
        # With a constraint, draw past n_samples so rejected draws can be replaced;
        # __next__ stops once n_samples were accepted, so the stream stays the same.
        n_draws = self.n_samples * (MAX_DRAWS_PER_SAMPLE if self.feasible is not None else 1)
        for _ in range(n_draws):
            quantiles = next(points)
            if boundary_rng.random() < self.boundary_fraction:
                quantiles = self._onto_boundary(quantiles, boundary_rng)
            yield [p.from_unit(q) for p, q in zip(parameters, quantiles.tolist(), strict=True)]

    def _unit_points(self, rng: np.random.Generator) -> Iterator[np.ndarray]:
        """Endless points in the unit cube, one coordinate per parameter."""
        d = self.n_inputs
        if self.picking_strategy == "random" or d == 0:
            while True:
                yield rng.random(d)
        from scipy.stats import qmc

        if self.picking_strategy == "sobol":
            engine = qmc.Sobol(d, scramble=True, rng=rng)
            # Sobol' points are balanced in blocks of a power of two: draw the first
            # 2**m >= n_samples, then double, so every prefix drawn is such a block.
            batch = engine.random(2 ** max(1, math.ceil(math.log2(max(self.n_samples, 1)))))
            while True:
                yield from batch
                batch = engine.random(engine.num_generated)
        else:  # lhs: one Latin hypercube of n_samples points after another
            engine = qmc.LatinHypercube(d, rng=rng)
            while True:
                yield from engine.random(max(self.n_samples, 1))

    def _onto_boundary(self, quantiles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``quantiles`` with 1 to all of its coordinates moved to 0 or 1, chosen at random."""
        moved = quantiles.copy()
        n_pinned = int(rng.integers(1, len(moved) + 1))
        pinned = rng.choice(len(moved), size=n_pinned, replace=False)
        moved[pinned] = rng.integers(0, 2, size=n_pinned)
        return moved
