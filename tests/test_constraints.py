"""Tests of the feasibility-constraint expressions and their agreement with is_feasible."""

from __future__ import annotations

import json

import pytest

from orca.geometry.constraints import (
    ConstraintError,
    evaluate_constraint,
    is_feasible_by,
    validate_constraint,
)
from orca.geometry.presets.inductor_octa import InductorOcta
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta


class TestEvaluator:
    @pytest.mark.parametrize(
        ("expression", "params", "expected"),
        [
            ("a <= b / 3", {"a": 1, "b": 3}, True),
            ("a <= b / 3", {"a": 1.1, "b": 3}, False),
            ("0 < a < b", {"a": 1, "b": 2}, True),
            ("0 < a < b", {"a": 2, "b": 1}, False),
            ("abs(a - b) <= 40 and a > 0", {"a": 10, "b": 45}, True),
            ("abs(a - b) <= 40 or a > 0", {"a": 10, "b": 60}, True),
            ("not a > b", {"a": 1, "b": 2}, True),
            ("(a if a > b else b) == 5", {"a": 5, "b": 2}, True),
            ("max(a, b) + min(a, b) == a + b", {"a": 3, "b": 7}, True),
            ("a / 2 * sin(radians(30)) > 0.99", {"a": 4}, True),
            ("sqrt(a) == sqrt2", {"a": 2}, True),
            ("ceil(100 * a) / 100 >= 1.24", {"a": 1.231}, True),
            ("-a < 0", {"a": 1}, True),
            ("a ** 2 % 3 == 1", {"a": 2}, True),
            ("a // 2 == 1", {"a": 3}, True),
            ("pi > 3", {}, True),
        ],
    )
    def test_evaluates(self, expression, params, expected):
        assert evaluate_constraint(expression, params) is expected

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('true')",
            "a.real > 0",
            "[a][0] > 0",
            "(lambda: 1)() > 0",
            "a if a else b",  # not boolean
            "exec('1') is None",
            "a is b",
            "a in (1, 2)",
            "f'{a}' == '1'",
            "pow(a, 2) > 0",  # not in FUNCTIONS
            "max(a, key=abs) > 0",
        ],
    )
    def test_rejects_unsupported_syntax(self, expression):
        with pytest.raises(ConstraintError):
            evaluate_constraint(expression, {"a": 1, "b": 2})

    def test_rejects_unknown_names(self):
        with pytest.raises(ConstraintError, match="'width', which is not an input parameter"):
            validate_constraint("width < diameter", ["diameter", "turns"])

    def test_rejects_non_boolean_result(self):
        with pytest.raises(ConstraintError, match="not to a boolean"):
            evaluate_constraint("a + b", {"a": 1, "b": 2})

    def test_rejects_syntax_errors(self):
        with pytest.raises(ConstraintError, match="not a valid expression"):
            validate_constraint("a <", ["a"])

    def test_is_feasible_by(self):
        assert is_feasible_by(["a > 0", "b > a"], {"a": 1, "b": 2})
        assert not is_feasible_by(["a > 0", "b > a"], {"a": 3, "b": 2})
        assert is_feasible_by([], {"a": 3})


@pytest.mark.parametrize("geometry_class", [TransformerOcta, InductorOcta])
class TestPresetConstraints:
    def test_are_valid_over_the_declared_inputs(self, geometry_class):
        geometry = geometry_class()
        names = list(geometry.input_parameter_iterator.get_ranges())
        for expression in geometry.feasibility_constraints():
            validate_constraint(expression, names)
        json.dumps(geometry.feasibility_constraints())

    def test_agree_with_is_feasible(self, geometry_class):
        # Random draws over the whole parameter box, without the constraint, so
        # both feasible and infeasible combinations are seen.
        geometry = geometry_class()
        iterator = geometry.input_parameter_iterator
        iterator.set_sample_count(2000, seed=9)
        constraints = geometry.feasibility_constraints()
        verdicts = set()
        for params in iterator:
            expected = geometry.is_feasible(params)
            assert is_feasible_by(constraints, params | {"frequency": 1e9}) is expected, params
            verdicts.add(expected)
        assert verdicts == {True, False}
