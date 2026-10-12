"""Errors shared by the public API."""


class OpenQMLError(Exception):
    """Base error for OpenQML."""


class ResourceLimitError(OpenQMLError, RuntimeError):
    """A requested simulation or readout exceeds an explicit resource budget."""


class NotFittedError(OpenQMLError, RuntimeError):
    """Prediction was requested before fitting a model."""


class UnsupportedOperationError(OpenQMLError, ValueError):
    """An operation is outside an explicitly supported subset."""
