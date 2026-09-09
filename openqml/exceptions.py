"""Exception hierarchy for OpenQML, mirroring the layout of openml.exceptions."""

__all__ = [
    "OpenQMLError",
    "OpenQMLServerError",
    "OpenQMLServerNoResult",
    "OpenQMLNotAuthorizedError",
    "OpenQMLCacheError",
    "ObjectNotPublishedError",
    "DuplicateRunError",
    "BackendNotAvailableError",
    "OpenQMLHashError",
]


class OpenQMLError(Exception):
    """Base class for all exceptions raised by openqml."""


class OpenQMLServerError(OpenQMLError):
    """The OpenQML server returned an unexpected status."""

    def __init__(self, message, code=None, url=None):
        self.code = code
        self.url = url
        super().__init__(message)


class OpenQMLServerNoResult(OpenQMLServerError):
    """The server (or local store) has no object with the requested identifier."""


class OpenQMLNotAuthorizedError(OpenQMLServerError):
    """The action requires an API key that is missing or rejected."""


class OpenQMLCacheError(OpenQMLError):
    """The on-disk cache is missing or corrupted."""


class ObjectNotPublishedError(OpenQMLError):
    """The entity has no id yet because it has never been published."""


class DuplicateRunError(OpenQMLError):
    """An identical run is already published.

    Its own class because bulk publishing wants to skip this one and only this
    one -- catching the base class there would swallow an auth or server
    failure and report success having uploaded nothing.
    """


class BackendNotAvailableError(OpenQMLError):
    """A quantum backend was requested but its dependencies are missing."""


class OpenQMLHashError(OpenQMLError):
    """A downloaded artefact did not match its recorded checksum."""
