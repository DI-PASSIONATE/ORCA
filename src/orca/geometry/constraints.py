"""
Feasibility constraints: closed-form rules between input parameters, as strings.

A geometry's :meth:`~orca.geometry.base_geometry.BaseGeometry.feasibility_constraints`
returns expressions such as ``"bottom_linewidth <= bottom_winding_diameter / 3"``.
They are written into the exported ONNX model under the ``input_constraints``
metadata key, so a consumer (COBRA) can tell an input it is about to query lies
outside the region the model was trained on, instead of getting a confident
prediction for a geometry that cannot be built.

The language is a subset of Python expressions and is evaluated here without
``eval``; a consumer in another language implements the same grammar:

- numbers, ``True``/``False``, and the input parameter names (including
  ``frequency``);
- ``+ - * / // % **``, unary ``-``, comparisons (chained allowed), ``and``,
  ``or``, ``not``, and the conditional ``a if c else b``;
- calls to :data:`FUNCTIONS`: ``abs min max sqrt sin cos tan radians ceil floor round``;
- the constants ``pi`` and ``sqrt2``.

Each expression must evaluate to a boolean; a parameter set is feasible when
every expression holds. Anything outside this grammar is rejected by
:func:`validate_constraint` at export time.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

#: Functions a constraint may call.
FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
}

#: Named constants a constraint may use.
CONSTANTS: Final[dict[str, float]] = {"pi": math.pi, "sqrt2": math.sqrt(2)}

_BINARY: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_COMPARE: Final[dict[type[ast.cmpop], Callable[[Any, Any], bool]]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class ConstraintError(ValueError):
    """A constraint expression is outside the supported grammar or names an unknown input."""


def validate_constraint(expression: str, input_names: list[str] | tuple[str, ...]) -> None:
    """Raise :class:`ConstraintError` unless *expression* is well-formed over *input_names*."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ConstraintError(f"Constraint {expression!r} is not a valid expression: {e.msg}") from e
    _check(tree.body, expression, set(input_names))


def evaluate_constraint(expression: str, params: Mapping[str, Any]) -> bool:
    """Whether *expression* holds for the parameter values in *params*.

    Raises:
        ConstraintError: If the expression is outside the grammar, names a
            parameter that is not in *params*, or does not evaluate to a boolean.
    """
    tree = ast.parse(expression, mode="eval")
    _check(tree.body, expression, set(params))
    result = _eval(tree.body, params)
    if not isinstance(result, bool):
        raise ConstraintError(f"Constraint {expression!r} evaluates to {result!r}, not to a boolean.")
    return result


def is_feasible_by(constraints: list[str], params: Mapping[str, Any]) -> bool:
    """Whether *params* satisfy every expression in *constraints*."""
    return all(evaluate_constraint(expression, params) for expression in constraints)


def _check(node: ast.expr, expression: str, names: set[str]) -> None:
    """Walk *node* and reject anything outside the grammar."""
    match node:
        case ast.Constant(value=bool() | int() | float()):
            pass
        case ast.Name(id=name):
            if name not in names and name not in CONSTANTS:
                raise ConstraintError(
                    f"Constraint {expression!r} uses {name!r}, which is not an input "
                    f"parameter ({', '.join(sorted(names))}) or a constant."
                )
        case ast.UnaryOp(op=ast.USub() | ast.UAdd() | ast.Not(), operand=operand):
            _check(operand, expression, names)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            _check(left, expression, names)
            _check(right, expression, names)
        case ast.BoolOp(values=values):
            for value in values:
                _check(value, expression, names)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            if any(type(op) not in _COMPARE for op in ops):
                raise ConstraintError(f"Constraint {expression!r} uses an unsupported comparison.")
            for value in (left, *comparators):
                _check(value, expression, names)
        case ast.IfExp(test=test, body=body, orelse=orelse):
            for value in (test, body, orelse):
                _check(value, expression, names)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
            if name not in FUNCTIONS:
                raise ConstraintError(
                    f"Constraint {expression!r} calls {name!r}; allowed functions are "
                    f"{', '.join(FUNCTIONS)}."
                )
            for value in args:
                _check(value, expression, names)
        case _:
            raise ConstraintError(
                f"Constraint {expression!r} contains unsupported syntax ({type(node).__name__})."
            )


def _eval(node: ast.expr, params: Mapping[str, Any]) -> Any:
    """Evaluate a node that passed :func:`_check`."""
    match node:
        case ast.Constant(value=value):
            return value
        case ast.Name(id=name):
            return params[name] if name in params else CONSTANTS[name]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand, params)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return +_eval(operand, params)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _eval(operand, params)
        case ast.BinOp(left=left, op=op, right=right):
            return _BINARY[type(op)](_eval(left, params), _eval(right, params))
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_eval(value, params) for value in values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return any(_eval(value, params) for value in values)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            current = _eval(left, params)
            for op, comparator in zip(ops, comparators, strict=True):
                following = _eval(comparator, params)
                if not _COMPARE[type(op)](current, following):
                    return False
                current = following
            return True
        case ast.IfExp(test=test, body=body, orelse=orelse):
            return _eval(body, params) if _eval(test, params) else _eval(orelse, params)
        case ast.Call(func=ast.Name(id=name), args=args):
            return FUNCTIONS[name](*(_eval(value, params) for value in args))
    raise ConstraintError(f"Unsupported node {type(node).__name__}.")  # unreachable after _check
