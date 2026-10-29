"""Serializable real expression graphs with native tensor evaluation and symbolic derivatives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np

from .errors import ValidationError


def real(value: Any) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and isfinite(float(value))
    except (ValueError, OverflowError, TypeError):
        valid = False
    if not valid:
        raise ValidationError(f"Expected a finite real scalar, got {value!r}.")
    return float(value)


def parameter_name(name):
    if not isinstance(name, str) or not name or any(ord(c) < 32 for c in name):
        raise ValidationError("Parameter names must be nonempty text without control characters.")
    return name


def _namespace(value):
    module = type(value).__module__
    if module.startswith("torch"):
        import torch

        return torch
    if module.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp
    return np


def _checked(value):
    if type(value) in {int, float}:
        if not isfinite(value):
            raise ValidationError("Parameter expression is undefined over finite real values.")
        return value
    if isinstance(value, complex):
        raise ValidationError("Parameter expressions must remain real.")
    if isinstance(value, (Real, np.ndarray, np.generic)):
        if np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            raise ValidationError("Parameter expression is undefined over finite real values.")
    return value


_ARITY = {
    "add": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
}


def _constant(value):
    # Older pickled expressions do not contain this derived constant flag.
    cached = getattr(value, "_parameter_free", None)
    return not value.parameters if cached is None else cached


def _postorder(roots, known=()):
    """Visit each immutable node once, without recursive hashing or tree expansion."""
    seen, pending = set(), [(value, False) for value in reversed(roots)]
    while pending:
        value, ready = pending.pop()
        identity = id(value)
        if identity in seen or identity in known:
            continue
        if ready or value.op is None:
            seen.add(identity)
            yield value
        else:
            pending.append((value, True))
            pending.extend((arg, False) for arg in reversed(value.args))


def _leaf_value(value, bindings):
    result = value.constant
    for name, coefficient in value.terms:
        if name not in bindings:
            raise ValidationError(f"Missing parameter binding: {name}.")
        result = result + coefficient * bindings[name]
    return _checked(result)


def _apply(op, args):
    try:
        if op == "add":
            result = args[0] + args[1]
        elif op == "mul":
            result = args[0] * args[1]
        elif op == "div":
            result = args[0] / args[1]
        elif op == "pow":
            result = args[0] ** args[1]
        else:
            result = getattr(_namespace(args[0]), op)(args[0])
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise ValidationError("Parameter expression is undefined over real values.") from exc
    return _checked(result)


def _evaluate_graph(value, bindings):
    nodes, uses = tuple(_postorder((value,))), {}
    for node in nodes:
        for arg in node.args:
            uses[id(arg)] = uses.get(id(arg), 0) + 1
    values = {}
    for node in nodes:
        values[id(node)] = (
            _leaf_value(node, bindings)
            if node.op is None
            else _apply(node.op, tuple(values[id(arg)] for arg in node.args))
        )
        for arg in node.args:
            key = id(arg)
            uses[key] -= 1
            if not uses[key]:
                del values[key]
    return values[id(value)]


def _evaluate(value, bindings):
    # Unary chains need neither graph indexing nor reference counts. A binary
    # tail still uses the shared evaluator, keeping arbitrarily deep paths flat.
    chain = []
    while value.op is not None and len(value.args) == 1:
        chain.append(value.op)
        value = value.args[0]
    result = _leaf_value(value, bindings) if value.op is None else _evaluate_graph(value, bindings)
    for op in reversed(chain):
        result = _apply(op, (result,))
    return result


def _rewrite_expression(value, mapping, *, substitute=False, memo=None):
    """Share rewrites across a circuit forest; replacement graphs remain simultaneous."""
    memo = {} if memo is None else memo
    if (existing := memo.get(id(value))) is not None:
        return existing
    for current in _postorder((value,), memo):
        if current.op is not None:
            args = tuple(memo[id(arg)] for arg in current.args)
            result = (
                current
                if not _constant(current)
                and all(a is b for a, b in zip(args, current.args, strict=True))
                else node(current.op, *args)
            )
        elif not any(name in mapping for name, _ in current.terms):
            result = current
        elif substitute:
            result = expression(current.constant)
            for name, coefficient in current.terms:
                result += coefficient * expression(mapping.get(name, Parameter(name)))
        else:
            result = Expression(
                tuple((k, v) for k, v in current.terms if k not in mapping),
                current.constant
                + sum(v * real(mapping[k]) for k, v in current.terms if k in mapping),
            )
        if current.op is None and result == current:
            result = current
        memo[id(current)] = result
    return memo[id(value)]


def _differentiate(value, name):
    derivatives, active = {}, {}
    for current in _postorder((value,)):
        key = id(current)
        if current.op is None:
            coefficient = dict(current.terms).get(name, 0)
            active[key], derivatives[key] = bool(coefficient), expression(coefficient)
            continue
        active[key] = any(active[id(arg)] for arg in current.args)
        if not active[key]:
            derivatives[key] = expression(0)
            continue
        a, da = current.args[0], derivatives[id(current.args[0])]
        if current.op == "sin":
            result = a.cos() * da
        elif current.op == "cos":
            result = -a.sin() * da
        elif current.op == "tan":
            result = da / a.cos() ** 2
        elif current.op == "exp":
            result = a.exp() * da
        elif current.op == "log":
            result = da / a
        elif current.op == "sqrt":
            result = da / (2 * a.sqrt())
        else:
            b, db = current.args[1], derivatives[id(current.args[1])]
            if current.op == "add":
                result = da + db
            elif current.op == "mul":
                result = da * b + a * db
            elif current.op == "div":
                result = (da * b - a * db) / b**2
            elif _constant(b):
                exponent = real(b.evaluate({}))
                result = exponent * a ** (exponent - 1) * da
            else:
                result = current * (db * a.log() + b * da / a)
        derivatives[key] = result
    return derivatives[id(value)]


@dataclass(frozen=True, eq=False)
class Expression:
    """Canonical affine terms and shared nonlinear graphs with native tensor evaluation.

    Evaluation, differentiation and rewrites visit each reachable node once.
    Evaluation releases temporary values after their final graph use. Numeric
    values and autodiff graphs are never cached between calls.
    """

    terms: tuple[tuple[str, float], ...] = ()
    constant: float = 0.0
    op: str | None = None
    args: tuple[Expression, ...] = ()

    def __post_init__(self):
        if self.op is not None:
            if self.op not in _ARITY or len(self.args) != _ARITY[self.op]:
                raise ValidationError("Unknown expression node or invalid arity.")
            if self.terms or self.constant != 0:
                raise ValidationError("A nonlinear node cannot also contain affine terms.")
            object.__setattr__(self, "args", tuple(expression(v) for v in self.args))
            object.__setattr__(self, "_parameter_free", all(_constant(v) for v in self.args))
            return
        if self.args:
            raise ValidationError("Affine expressions cannot contain graph arguments.")
        combined = {}
        for name, coefficient in self.terms:
            name, coefficient = parameter_name(name), real(coefficient)
            combined[name] = combined.get(name, 0) + coefficient
        object.__setattr__(self, "constant", real(self.constant))
        object.__setattr__(
            self, "terms", tuple((k, real(v)) for k, v in sorted(combined.items()) if v)
        )
        object.__setattr__(self, "_parameter_free", not self.terms)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        if self is other:
            return True
        if (
            self.terms != other.terms
            or self.constant != other.constant
            or self.op != other.op
            or len(self.args) != len(other.args)
        ):
            return False
        if self.op is None:
            return True
        pending, seen = [(self, other)], set()
        while pending:
            left, right = pending.pop()
            key = id(left), id(right)
            if left is right or key in seen:
                continue
            if (
                left.__class__ is not right.__class__
                or left.terms != right.terms
                or left.constant != right.constant
                or left.op != right.op
                or len(left.args) != len(right.args)
            ):
                return False
            seen.add(key)
            pending.extend(zip(left.args, right.args, strict=True))
        return True

    def __hash__(self):
        if self.op is None:
            return hash((self.terms, self.constant, self.op, ()))
        hashes = {}
        for value in _postorder((self,)):
            hashes[id(value)] = hash(
                (value.terms, value.constant, value.op, tuple(hashes[id(a)] for a in value.args))
            )
        return hashes[id(self)]

    @property
    def parameters(self):
        if self.op is None:
            return frozenset(k for k, _ in self.terms)
        names, seen, pending = set(), set(), [self]
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.op is None:
                names.update(k for k, _ in value.terms)
            else:
                pending.extend(value.args)
        return frozenset(names)

    @property
    def is_affine(self):
        return self.op is None

    def evaluate(self, bindings: Mapping[str, Any]):
        if self.op is None:
            return _leaf_value(self, bindings)
        if all(arg.op is None for arg in self.args):
            first = _leaf_value(self.args[0], bindings)
            args = (first,)
            if len(self.args) == 2:
                second = (
                    first if self.args[1] is self.args[0] else _leaf_value(self.args[1], bindings)
                )
                args = first, second
            return _apply(self.op, args)
        return _evaluate(self, bindings)

    def bind(self, bindings):
        return _rewrite_expression(self, bindings)

    def derivative(self, name):
        return _differentiate(self, name)

    def substitute(self, mapping):
        return _rewrite_expression(self, mapping, substitute=True)

    def to_dict(self):
        if self.op is None:
            return {"constant": self.constant, "terms": dict(self.terms)}
        return {"op": self.op, "args": [arg.to_dict() for arg in self.args]}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValidationError("Expression must be an object.")
        if set(data) == {"constant", "terms"} and isinstance(data["terms"], dict):
            return cls(tuple(data["terms"].items()), data["constant"])
        if set(data) == {"op", "args"} and isinstance(data["args"], list):
            return node(data["op"], *(cls.from_dict(v) for v in data["args"]))
        raise ValidationError("Invalid expression fields.")

    def __add__(self, other):
        other = expression(other)
        if self.is_affine and other.is_affine:
            return Expression(self.terms + other.terms, self.constant + other.constant)
        return node("add", self, other)

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = expression(other)
        return expression(0) if self == other else self + -other

    def __rsub__(self, other):
        return expression(other) - self

    def __mul__(self, other):
        other = expression(other)
        if self.is_affine and other.is_affine and (not self.terms or not other.terms):
            if other.terms:
                return other * self.constant
            return Expression(
                tuple((k, v * other.constant) for k, v in self.terms),
                self.constant * other.constant,
            )
        return node("mul", self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = expression(other)
        if _constant(other):
            value = real(other.evaluate({}))
            if value == 0:
                raise ValidationError("Division by zero.")
            return self * (1 / value)
        return node("div", self, other)

    def __rtruediv__(self, other):
        return expression(other) / self

    def __pow__(self, other):
        return node("pow", self, expression(other))

    def __rpow__(self, other):
        return node("pow", expression(other), self)

    def sin(self):
        return node("sin", self)

    def cos(self):
        return node("cos", self)

    def tan(self):
        return node("tan", self)

    def exp(self):
        return node("exp", self)

    def log(self):
        return node("log", self)

    def sqrt(self):
        return node("sqrt", self)


def node(op, *args):
    if op not in _ARITY or len(args) != _ARITY[op]:
        raise ValidationError("Unknown expression node or invalid arity.")
    args = tuple(expression(arg) for arg in args)
    if all(_constant(arg) for arg in args):
        return expression(Expression(op=op, args=args).evaluate({}))
    if op == "add":
        if args[0] == expression(0):
            return args[1]
        if args[1] == expression(0):
            return args[0]
    if op == "mul":
        if expression(0) in args:
            return expression(0)
        if args[0] == expression(1):
            return args[1]
        if args[1] == expression(1):
            return args[0]
    if op == "pow":
        if args[1] == expression(0):
            return expression(1)
        if args[1] == expression(1):
            return args[0]
    return Expression(op=op, args=args)


@dataclass(frozen=True)
class Parameter:
    name: str

    def __post_init__(self):
        parameter_name(self.name)

    def __add__(self, other):
        return expression(self) + other

    __radd__ = __add__

    def __neg__(self):
        return -expression(self)

    def __sub__(self, other):
        return expression(self) - other

    def __rsub__(self, other):
        return expression(other) - self

    def __mul__(self, other):
        return expression(self) * other

    __rmul__ = __mul__

    def __truediv__(self, other):
        return expression(self) / other

    def __rtruediv__(self, other):
        return expression(other) / self

    def __pow__(self, other):
        return expression(self) ** other

    def __rpow__(self, other):
        return expression(other) ** expression(self)

    def sin(self):
        return expression(self).sin()

    def cos(self):
        return expression(self).cos()

    def tan(self):
        return expression(self).tan()

    def exp(self):
        return expression(self).exp()

    def log(self):
        return expression(self).log()

    def sqrt(self):
        return expression(self).sqrt()


def expression(value):
    if isinstance(value, Expression):
        return value
    if isinstance(value, Parameter):
        return Expression(((value.name, 1.0),))
    return Expression(constant=real(value))
