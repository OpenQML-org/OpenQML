"""Runtime configuration for openqml.

The module mirrors ``openml.config``: values are plain module-level globals, so
they can be set directly::

    import openqml
    openqml.config.server = "https://example.org/api/v1"
    openqml.config.apikey = "..."

The default server is the sentinel ``"local"``. In that mode every call is
served by an on-disk store under the cache directory, so the whole library --
datasets, tasks, runs, the leaderboard -- works with no network and no account.
Point ``server`` at a real deployment to talk to it over HTTP.
"""

from __future__ import annotations

import logging
import os
import pathlib

__all__ = [
    "server",
    "apikey",
    "connection_n_retries",
    "avoid_duplicate_runs",
    "show_progress",
    "LOCAL_SERVER",
    "get_cache_directory",
    "set_root_cache_directory",
    "get_root_cache_directory",
    "get_config_as_dict",
    "set_server",
    "use_local_server",
    "is_local",
    "logger",
]

LOCAL_SERVER = "local"

#: Either ``"local"`` or the base URL of an OpenQML REST API.
server: str = os.environ.get("OPENQML_SERVER", LOCAL_SERVER)

#: API key used for write operations against a remote server.
apikey: str = os.environ.get("OPENQML_APIKEY", "")

#: How often a failed HTTP request is retried.
connection_n_retries: int = 3

#: If True, ``runs.run_model_on_task`` refuses to recompute an identical run.
avoid_duplicate_runs: bool = True

#: Print progress information during long-running local computations.
show_progress: bool = False

_root_cache_directory = pathlib.Path(
    os.environ.get("OPENQML_CACHE_DIR", pathlib.Path.home() / ".cache" / "openqml")
).expanduser()

logger = logging.getLogger("openqml")


def get_root_cache_directory() -> pathlib.Path:
    """Return the root of the on-disk cache."""
    return _root_cache_directory


def set_root_cache_directory(path) -> None:
    """Point the cache (and the local store) at another directory."""
    global _root_cache_directory
    _root_cache_directory = pathlib.Path(path).expanduser()
    _root_cache_directory.mkdir(parents=True, exist_ok=True)


def get_cache_directory(*parts) -> pathlib.Path:
    """Return (and create) a sub-directory of the cache, e.g. ``("datasets", "3")``.

    Every part is a single name, never a path. Entity ids and names reach here
    from stored or downloaded JSON, and ``Path.joinpath`` treats an absolute
    part as a fresh root -- ``joinpath("datasets", "/etc/x")`` is ``/etc/x`` --
    so an unchecked part would let a payload choose where the cache is written.
    """
    safe = []
    for part in parts:
        name = str(part)
        if not name or name in (".", "..") or "/" in name or "\\" in name or os.path.isabs(name):
            raise ValueError(
                f"invalid cache path component {name!r}: a component must be a single "
                f"name, not a path"
            )
        safe.append(name)
    path = _root_cache_directory.joinpath(*safe)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_local() -> bool:
    """True when calls are served by the on-disk store instead of HTTP."""
    return server == LOCAL_SERVER


def set_server(url: str) -> None:
    """Set the API base URL (``"local"`` switches back to the on-disk store)."""
    global server
    server = url


def use_local_server() -> None:
    """Switch back to the bundled on-disk store."""
    set_server(LOCAL_SERVER)


def get_config_as_dict() -> dict:
    """Return the active configuration -- the API key is never included verbatim."""
    return {
        "server": server,
        "apikey": ("<set>" if apikey else ""),
        "cache_directory": str(_root_cache_directory),
        "connection_n_retries": connection_n_retries,
        "avoid_duplicate_runs": avoid_duplicate_runs,
    }
