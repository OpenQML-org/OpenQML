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


@dataclass(frozen=True)
class Expression:
    """Affine expressions retain canonical terms; nonlinear graphs use whitelisted nodes."""

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
            result = self.constant
            for name, coefficient in self.terms:
                if name not in bindings:
                    raise ValidationError(f"Missing parameter binding: {name}.")
                result = result + coefficient * bindings[name]
            return _checked(result)
        args = tuple(arg.evaluate(bindings) for arg in self.args)
        try:
            if self.op == "add":
                result = args[0] + args[1]
            elif self.op == "mul":
                result = args[0] * args[1]
            elif self.op == "div":
                result = args[0] / args[1]
            elif self.op == "pow":
                result = args[0] ** args[1]
            else:
                result = getattr(_namespace(args[0]), self.op)(args[0])
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise ValidationError("Parameter expression is undefined over real values.") from exc
        return _checked(result)

    def bind(self, bindings):
        if self.op is not None:
            return node(self.op, *(arg.bind(bindings) for arg in self.args))
        return Expression(
            tuple((k, v) for k, v in self.terms if k not in bindings),
            self.constant + sum(v * real(bindings[k]) for k, v in self.terms if k in bindings),
        )

    def derivative(self, name):
        if name not in self.parameters:
            return expression(0)
        if self.op is None:
            return expression(dict(self.terms).get(name, 0))
        a = self.args[0]
        da = a.derivative(name)
        if self.op == "sin":
            return a.cos() * da
        if self.op == "cos":
            return -a.sin() * da
        if self.op == "tan":
            return da / a.cos() ** 2
        if self.op == "exp":
            return a.exp() * da
        if self.op == "log":
            return da / a
        if self.op == "sqrt":
            return da / (2 * a.sqrt())
        b, db = self.args[1], self.args[1].derivative(name)
        if self.op == "add":
            return da + db
        if self.op == "mul":
            return da * b + a * db
        if self.op == "div":
            return (da * b - a * db) / b**2
        if not b.parameters:
            exponent = real(b.evaluate({}))
            return exponent * a ** (exponent - 1) * da
        return self * (db * a.log() + b * da / a)

    def substitute(self, mapping):
        if self.op is not None:
            return node(self.op, *(arg.substitute(mapping) for arg in self.args))
        result = expression(self.constant)
        for name, coefficient in self.terms:
            result += coefficient * expression(mapping.get(name, Parameter(name)))
        return result

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
        if not other.parameters:
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
    if not any(arg.parameters for arg in args):
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
