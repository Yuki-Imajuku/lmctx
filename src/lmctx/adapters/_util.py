"""Shared utilities for adapter implementations."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmctx.plan import AdapterId
    from lmctx.spec import RunSpec


def _as_object_dict(value: object) -> dict[str, object]:
    """Normalize a mapping-like object into dict[str, object]."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): val for key, val in value.items()}


def _to_dict(obj: object) -> dict[str, Any]:
    """Convert a provider SDK response object or dict to a plain dict.

    Supports:
    - Plain dicts (returned as-is)
    - Pydantic models (via model_dump)
    - Objects with to_dict method
    """
    if isinstance(obj, Mapping):
        return {str(key): value for key, value in obj.items()}

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        dumped: object
        try:
            dumped = model_dump(mode="python", warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return {str(key): value for key, value in dumped.items()}

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return {str(key): value for key, value in dumped.items()}

    msg = f"Cannot convert {type(obj).__name__} to dict; expected dict or object with model_dump()/to_dict()"
    raise TypeError(msg)


def _deep_merge(base: dict[str, object], override: Mapping[str, object]) -> dict[str, object]:
    """Deep-merge override into base (non-mutating).

    Dicts are recursively merged; all other values are replaced by override.
    """
    result: dict[str, object] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(_as_object_dict(result[key]), _as_object_dict(value))
        else:
            result[key] = value
    return result


def _to_json_compatible(value: object) -> object:
    """Convert mapping/tuple containers into JSON-serializable structures."""
    if isinstance(value, Enum):
        return _to_json_compatible(value.value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    return value


def _json_object(mapping: Mapping[str, object]) -> dict[str, object]:
    """Normalize a mapping into a JSON-compatible ``dict[str, object]``."""
    return {str(key): _to_json_compatible(value) for key, value in mapping.items()}


def _validate_adapter_spec(adapter_id: AdapterId, spec: RunSpec) -> None:
    """Validate that RunSpec targets this adapter's provider/endpoint."""
    if spec.provider != adapter_id.provider:
        msg = (
            f"RunSpec provider mismatch: expected '{adapter_id.provider}' for "
            f"{adapter_id.provider}/{adapter_id.endpoint}, got '{spec.provider}'."
        )
        raise ValueError(msg)
    if spec.endpoint != adapter_id.endpoint:
        msg = (
            f"RunSpec endpoint mismatch: expected '{adapter_id.endpoint}' for "
            f"{adapter_id.provider}/{adapter_id.endpoint}, got '{spec.endpoint}'."
        )
        raise ValueError(msg)
    if adapter_id.api_version is not None and spec.api_version != adapter_id.api_version:
        msg = (
            f"RunSpec api_version mismatch: expected '{adapter_id.api_version}' for "
            f"{adapter_id.provider}/{adapter_id.endpoint}, got '{spec.api_version}'."
        )
        raise ValueError(msg)


def _plan_extra_hints(spec: RunSpec) -> dict[str, object]:
    """Return non-request client hints surfaced through ``RequestPlan.extra``."""
    extra: dict[str, object] = {}
    if spec.base_url:
        extra["base_url"] = spec.base_url
    if spec.api_version:
        extra["api_version"] = spec.api_version
    return extra
