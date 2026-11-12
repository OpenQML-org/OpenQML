"""Structured failures shared by the IR, simulators, and adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T")


class OpenQMLError(Exception):
    """Base class for public OpenQML exceptions."""


class ValidationError(OpenQMLError):
    """Invalid circuit, parameters, or runtime options."""


def _require_instance(value: object, expected: type[_T], name: str) -> _T:
    """Validate public input types before dispatch, without coercing user data."""
    if not isinstance(value, expected):
        raise ValidationError(
            f"{name} must be a {expected.__name__} instance; got {type(value).__name__}."
        )
    return value


def _require_mapping(value, name="bindings"):
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping; got {type(value).__name__}.")
    return value


def _require_choice(value, choices, name):
    _require_instance(value, str, name)
    if value not in choices:
        shown = repr(value[:80]) + ("..." if len(value) > 80 else "")
        raise ValidationError(f"{name} must be one of {tuple(choices)}; got {shown}.")
    return value


def _as_tuple(value, name):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValidationError(f"{name} must be an iterable of items; got {type(value).__name__}.")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise ValidationError(
            f"{name} must be an iterable of items; got {type(value).__name__}."
        ) from exc
    return tuple(iterator)


class ResourceError(OpenQMLError):
    """Execution exceeds a configured resource budget."""


class MissingDependencyError(OpenQMLError):
    """An integration dependency could not be imported."""


class BackendError(OpenQMLError):
    """An execution backend failed or returned an invalid result."""


class NumericalError(BackendError):
    """A numerical primitive failed or produced a non-finite result."""


class ConvergenceError(NumericalError):
    """An iterative solution failed its freshly recomputed residual contract."""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    location: str = "circuit"
    severity: str = "error"
    hint: str = ""

    def __str__(self) -> str:
        suffix = f" Hint: {self.hint}" if self.hint else ""
        return f"[{self.code}] {self.location}: {self.message}{suffix}"


class ConversionError(OpenQMLError):
    def __init__(self, diagnostics: list[Diagnostic] | tuple[Diagnostic, ...]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(str(d) for d in self.diagnostics))

    def __reduce__(self):
        return type(self), (self.diagnostics,)


def conversion_error(code: str, message: str, location: str = "circuit", hint: str = ""):
    return ConversionError([Diagnostic(code, message, location, hint=hint)])
