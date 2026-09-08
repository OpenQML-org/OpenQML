"""Flow-level API."""

from __future__ import annotations

from typing import Optional, Sequence

from .._store import get_store
from ..utils import as_table, get_params, jsonify, stable_hash
from .flow import OpenQMLFlow

__all__ = ["model_to_flow", "flow_to_model", "get_flow", "list_flows", "flow_exists",
           "publish_flow"]


def _version_of(model) -> str:
    module = type(model).__module__.split(".")[0]
    try:
        package = __import__(module)
        version = getattr(package, "__version__", "unknown")
    except Exception:  # pragma: no cover - defensive
        version = "unknown"
    return f"{module}=={version}"


def model_to_flow(model) -> OpenQMLFlow:
    """Describe any estimator (built-in, scikit-learn, or your own) as a flow."""
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    parameters = jsonify(get_params(model))
    external_version = _version_of(model)
    return OpenQMLFlow(
        name=f"{type(model).__name__}({stable_hash(parameters)[:8]})",
        model_class=model_class,
        parameters=parameters,
        external_version=external_version,
        description=(type(model).__doc__ or "").strip().split("\n")[0],
        dependencies=external_version,
    )


def flow_to_model(flow, **overrides):
    """Inverse of :func:`model_to_flow`."""
    if isinstance(flow, (int, str)):
        flow = get_flow(flow)
    return flow.to_model(**overrides)


def get_flow(identifier) -> OpenQMLFlow:
    return OpenQMLFlow.from_dict(get_store().get("flows", identifier))


def list_flows(tag: Optional[str] = None, output_format: str = "dataframe"):
    records = []
    for payload in get_store().list("flows"):
        if tag and tag not in payload.get("tags", []):
            continue
        records.append({
            "id": payload.get("id"),
            "name": payload.get("name"),
            "model_class": payload.get("model_class"),
            "external_version": payload.get("external_version"),
        })
    return as_table(records, output_format, index="id")


def flow_exists(flow: OpenQMLFlow) -> Optional[int]:
    """Return the id of an identical flow (same class, version, parameters)."""
    target = stable_hash({"class": flow.model_class, "version": flow.external_version,
                          "parameters": flow.parameters})
    for payload in get_store().list("flows"):
        existing = stable_hash({"class": payload.get("model_class"),
                                "version": payload.get("external_version"),
                                "parameters": payload.get("parameters", {})})
        if existing == target:
            return int(payload["id"])
    return None


def publish_flow(flow: OpenQMLFlow) -> OpenQMLFlow:
    """Publish a flow, reusing an identical one if it is already registered."""
    existing = flow_exists(flow)
    if existing is not None:
        flow.id = existing
        return flow
    return flow.publish()
