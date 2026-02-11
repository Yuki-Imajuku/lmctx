"""Core data types: Part, Message, ToolSpecification, Cursor, Usage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lmctx.blobs import BlobReference

Role = Literal["system", "developer", "user", "assistant", "tool"]


def _freeze_value(value: object) -> object:
    """Freeze nested payload containers to keep snapshots immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """Freeze a mapping into a read-only mappingproxy."""
    if value is None:
        return None
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


@dataclass(frozen=True, slots=True)
class Part:
    """A single content block within a Message."""

    # Known values include text/json/image/audio/file/tool_call/tool_result/thinking/compaction.
    type: str
    text: str | None = None
    json: Mapping[str, object] | None = None
    blob: BlobReference | None = None

    # tool fields
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: Mapping[str, object] | None = None
    tool_output: object | None = None

    # provider raw for lossless debugging (optional)
    provider_raw: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Freeze mutable payload fields to keep snapshots immutable."""
        object.__setattr__(self, "json", _freeze_mapping(self.json))
        object.__setattr__(self, "tool_args", _freeze_mapping(self.tool_args))
        object.__setattr__(self, "tool_output", _freeze_value(self.tool_output))
        object.__setattr__(self, "provider_raw", _freeze_mapping(self.provider_raw))


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in the conversation log: role + parts."""

    role: Role
    parts: tuple[Part, ...]

    id: str | None = None
    provider: str | None = None
    turn_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Normalize parts container to tuple for runtime safety."""
        object.__setattr__(self, "parts", tuple(self.parts))


@dataclass(frozen=True, slots=True)
class ToolSpecification:
    """JSON-Schema-based tool definition."""

    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze the schema mapping to avoid external mutation."""
        frozen = _freeze_mapping(self.input_schema)
        if frozen is None:
            msg = "input_schema cannot be None."
            raise TypeError(msg)
        object.__setattr__(self, "input_schema", frozen)


@dataclass(frozen=True, slots=True)
class Cursor:
    """Provider-specific state for conversation chaining."""

    last_response_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage statistics from a single LLM call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_usage: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Freeze provider usage payload."""
        frozen = _freeze_mapping(self.provider_usage)
        if frozen is None:
            msg = "provider_usage cannot be None."
            raise TypeError(msg)
        object.__setattr__(self, "provider_usage", frozen)
