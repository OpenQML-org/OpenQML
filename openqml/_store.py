"""Transport layer: an on-disk store and a thin REST client.

Everything above this module speaks in dicts and never knows whether the bytes
came from disk or from an HTTP endpoint. ``config.server == "local"`` selects
the on-disk store, anything else selects the REST client.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import config
from .exceptions import (
    OpenQMLNotAuthorizedError,
    OpenQMLServerError,
    OpenQMLServerNoResult,
)
from .utils import jsonify

_LOCK = threading.RLock()
_ENTITIES = ("datasets", "tasks", "flows", "runs", "suites")


class LocalStore:
    """A file-backed stand-in for the OpenQML server.

    Bundled entities are seeded from :mod:`openqml.catalog` the first time the
    store is touched; anything published afterwards is written next to them as
    JSON, so ids and the leaderboard survive across processes.
    """

    def __init__(self, root=None):
        self.root = (root or config.get_root_cache_directory()) / "local_store"
        self._seeded = False

    # -- plumbing ---------------------------------------------------------
    def _dir(self, entity: str):
        path = self.root / entity
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _seed(self) -> None:
        if self._seeded:
            return
        with _LOCK:
            from .catalog import SEED_ENTITIES

            for entity, objects in SEED_ENTITIES.items():
                folder = self._dir(entity)
                for obj in objects:
                    target = folder / f"{obj['id']}.json"
                    if not target.exists():
                        target.write_text(json.dumps(obj, indent=2), encoding="utf-8")
            self._seeded = True

    def _next_id(self, entity: str) -> int:
        self._seed()
        existing = [int(p.stem) for p in self._dir(entity).glob("*.json") if p.stem.isdigit()]
        return max(existing, default=0) + 1

    # -- api --------------------------------------------------------------
    def get(self, entity: str, identifier) -> Dict[str, Any]:
        self._seed()
        folder = self._dir(entity)
        if isinstance(identifier, str) and not identifier.isdigit():
            for record in self.list(entity):
                if record.get("name") == identifier or record.get("alias") == identifier:
                    return record
            raise OpenQMLServerNoResult(
                f"no {entity[:-1]} named {identifier!r} in the local store"
            )
        path = folder / f"{int(identifier)}.json"
        if not path.exists():
            raise OpenQMLServerNoResult(f"{entity[:-1]} {identifier} not found in the local store")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, entity: str, **filters) -> List[Dict[str, Any]]:
        self._seed()
        records = []
        for path in sorted(self._dir(entity).glob("*.json"), key=lambda p: int(p.stem)):
            record = json.loads(path.read_text(encoding="utf-8"))
            if all(record.get(k) == v for k, v in filters.items() if v is not None):
                records.append(record)
        return records

    def post(self, entity: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._seed()
        with _LOCK:
            payload = dict(jsonify(payload))
            identifier = payload.get("id") or self._next_id(entity)
            payload["id"] = int(identifier)
            payload.setdefault("uploaded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            (self._dir(entity) / f"{payload['id']}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        return payload

    def delete(self, entity: str, identifier) -> None:
        path = self._dir(entity) / f"{int(identifier)}.json"
        if path.exists():
            path.unlink()

    def reset(self) -> None:
        """Drop every locally published entity and re-seed the bundled ones."""
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)
        self._seeded = False
        self._seed()


class RestStore:
    """Minimal JSON client for an OpenQML deployment (stdlib only)."""

    def __init__(self, base_url: str, apikey: str = ""):
        self.base_url = base_url.rstrip("/")
        self.apikey = apikey

    def _request(self, method: str, path: str, payload=None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = None
        headers = {"Accept": "application/json", "User-Agent": "openqml-python"}
        if payload is not None:
            data = json.dumps(jsonify(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.apikey:
            headers["Authorization"] = f"Bearer {self.apikey}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        last_error: Optional[Exception] = None
        for attempt in range(max(1, config.connection_n_retries)):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise OpenQMLServerNoResult(f"{url} returned 404", code=404, url=url) from None
                if exc.code in (401, 403):
                    raise OpenQMLNotAuthorizedError(
                        "the server rejected the API key; set openqml.config.apikey",
                        code=exc.code,
                        url=url,
                    ) from None
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
            time.sleep(0.5 * (attempt + 1))
        raise OpenQMLServerError(f"request to {url} failed: {last_error}", url=url)

    def get(self, entity: str, identifier) -> Dict[str, Any]:
        return self._request("GET", f"/{entity}/{identifier}")

    def list(self, entity: str, **filters) -> List[Dict[str, Any]]:
        query = "&".join(f"{k}={v}" for k, v in filters.items() if v is not None)
        suffix = f"?{query}" if query else ""
        payload = self._request("GET", f"/{entity}{suffix}")
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    def post(self, entity: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.apikey:
            raise OpenQMLNotAuthorizedError("publishing requires openqml.config.apikey")
        return self._request("POST", f"/{entity}", payload)

    def delete(self, entity: str, identifier) -> None:
        self._request("DELETE", f"/{entity}/{identifier}")


_LOCAL_SINGLETON: Optional[LocalStore] = None


def get_store():
    """Return the store selected by :mod:`openqml.config`."""
    global _LOCAL_SINGLETON
    if config.is_local():
        if _LOCAL_SINGLETON is None or _LOCAL_SINGLETON.root.parent != config.get_root_cache_directory():
            _LOCAL_SINGLETON = LocalStore()
        return _LOCAL_SINGLETON
    return RestStore(config.server, config.apikey)
