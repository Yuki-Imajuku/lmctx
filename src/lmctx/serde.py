"""Shared serialization and validation utilities for to_dict / from_dict round-trips."""

from collections.abc import Mapping


def to_plain_data(value: object) -> object:
    """Recursively normalize Mapping/tuple containers into plain dict/list values."""
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_plain_data(item) for item in value]
    return value


def as_str_object_dict(value: object, *, field_name: str) -> dict[str, object]:
    """Validate and normalize a mapping value into ``dict[str, object]``."""
    if not isinstance(value, Mapping):
        msg = f"{field_name} must be a mapping."
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def optional_string(value: object, *, field_name: str) -> str | None:
    """Validate an optional string field."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{field_name} must be a string or None."
        raise TypeError(msg)
    return value


def optional_int(value: object, *, field_name: str) -> int | None:
    """Validate an optional integer field (rejects booleans)."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{field_name} must be an int or None."
        raise TypeError(msg)
    return value


def require_int(value: object, *, field_name: str) -> int:
    """Validate a required integer field (rejects booleans)."""
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{field_name} must be an int."
        raise TypeError(msg)
    return value


def optional_float(value: object, *, field_name: str) -> float | None:
    """Validate an optional float field (rejects booleans)."""
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        msg = f"{field_name} must be a float or None."
        raise TypeError(msg)
    return float(value)


def string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Validate and normalize an optional sequence of strings into ``tuple[str, ...]``."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        msg = f"{field_name} must be a sequence of strings."
        raise TypeError(msg)

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            msg = f"{field_name}[{index}] must be a string."
            raise TypeError(msg)
        result.append(item)
    return tuple(result)
