"""Structured failures shared by the IR, simulators, and adapters."""

from dataclasses import dataclass


class OpenQMLError(Exception):
    """Base class for public OpenQML exceptions."""


class ValidationError(OpenQMLError, ValueError):
    """Invalid circuit, parameters, or runtime options."""


class ResourceError(OpenQMLError, MemoryError):
    """Execution exceeds a configured resource budget."""


class MissingDependencyError(OpenQMLError, ImportError):
    """An optional integration was requested without its dependencies."""


class BackendError(OpenQMLError, RuntimeError):
    """An execution backend failed or returned an invalid result."""


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


class ConversionError(OpenQMLError, ValueError):
    def __init__(self, diagnostics: list[Diagnostic] | tuple[Diagnostic, ...]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("\n".join(str(d) for d in self.diagnostics))


def conversion_error(code: str, message: str, location: str = "circuit", hint: str = ""):
    return ConversionError([Diagnostic(code, message, location, hint=hint)])
